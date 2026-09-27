"""Qwen-DANN (laptop #177): continue the trained Qwen3-4B LoRA adapter with a gradient-reversal domain head, so the
pooled representation of unlabelled FRANCE pairs aligns with labelled India/US pairs. No France labels are used.

source : labelled fold != 0 train pairs (inputs/train.parquet), match loss (BCE) on source only
target : unlabelled France test pairs (inputs/france_all.parquet), domain loss only
pooled : the last non-pad token's final hidden state (the vector Qwen's score head reads)
lambda : Ganin ramp 2/(1+exp(-10p)) - 1, times --max_lambda. Domain head: MLP -> 256 -> 1, dropout 0.1, 10x lr.
Output : <out>/adapter (PEFT adapter incl. score head) + DONE; score it with qwen_score.py --adapter <out>/adapter.
Only run after (1) plain Qwen helps France on the LB and (2) the xlm-r DANN proxy passes laptop #167.
"""
import argparse, math, os, time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

p = argparse.ArgumentParser()
p.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-4B-Base"))
p.add_argument("--adapter", default=os.path.expanduser("~/work/qwen_ce/out/adapter"))
p.add_argument("--inputs", default=os.path.expanduser("~/work/qwen_ce/inputs"))
p.add_argument("--data", default=os.path.expanduser("~/work/dataset/dataset"))
p.add_argument("--target", default="france_all.parquet")
p.add_argument("--out", default=os.path.expanduser("~/work/qwen_ce/out_dann"))
p.add_argument("--n_src", type=int, default=300_000)
p.add_argument("--src_file", default="", help="optional parquet of labelled TEXT pairs (text_a, text_b, label) used as the source instead of train.parquet ids (self-training: France pseudo-labels + IN/US replay)")
p.add_argument("--bs", type=int, default=32, help="labelled source pairs per step (plus as many target pairs)")
p.add_argument("--accum", type=int, default=2)
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--max_lambda", type=float, default=0.1)
p.add_argument("--dom_lr_mult", type=float, default=10.0)
p.add_argument("--max_len", type=int, default=128)
p.add_argument("--save_every", type=int, default=1000)
p.add_argument("--seed", type=int, default=11)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
t0 = time.time()
log = lambda *a: print(f"[{(time.time() - t0) / 60:7.1f} min]", *a, flush=True)
dev = torch.device("cuda")
_reserve = torch.empty(int(40 * 2**30), dtype=torch.uint8, device=dev); del _reserve  # hold room on a shared GPU


def texts(split):
    out = {}
    for s in (1, 2, 3):
        df = pd.read_csv(f"{args.data}/{split}/{split}_source{s}.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
        out.update(zip(df.entity_id, df.business_name + " | " + df.business_address))
    return out


tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = "<|endoftext|>"
enc = lambda a, b: tok([f"Record A: {x}\nRecord B: {y}\nSame business?" for x, y in zip(a, b)], truncation=True,
                       max_length=args.max_len, add_special_tokens=False)["input_ids"]

if args.src_file:
    src = pd.read_parquet(args.src_file).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    src_ids = enc(list(src.text_a), list(src.text_b))
else:
    src = pd.read_parquet(os.path.join(args.inputs, "train.parquet")).sample(args.n_src, random_state=args.seed)
    lut = texts("train")
    src_ids = enc([lut[x] for x in src.s1_id], [lut[x] for x in src.pool_id]); del lut
tgt = pd.read_parquet(os.path.join(args.inputs, args.target))
lut = texts("test")
tgt_ids = enc([lut[x] for x in tgt.s1_id], [lut[x] for x in tgt.pool_id]); del lut
y_all = torch.tensor(src.label.values, dtype=torch.float32)
log(f"source {len(src_ids):,} (pos {src.label.mean():.3f}) | target {len(tgt_ids):,} unlabelled")


def collate(ids):
    L = max(len(x) for x in ids)
    inp = torch.full((len(ids), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(ids), L), dtype=torch.long)
    for i, x in enumerate(ids):
        inp[i, : len(x)] = torch.tensor(x); att[i, : len(x)] = 1
    return inp.to(dev), att.to(dev)


base = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1, dtype=torch.bfloat16,
                                                          attn_implementation="sdpa")
base.config.pad_token_id = tok.pad_token_id
model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True).to(dev)
H = base.config.hidden_size


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


dom = torch.nn.Sequential(torch.nn.Linear(H, 256), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, 1)).to(dev)
params = [q for q in model.parameters() if q.requires_grad]
opt = torch.optim.AdamW([{"params": params, "lr": args.lr}, {"params": dom.parameters(), "lr": args.lr * args.dom_lr_mult}],
                        weight_decay=0.01)
steps = math.ceil(len(src_ids) / args.bs)
sch = get_linear_schedule_with_warmup(opt, int(0.03 * steps), steps)


def pooled_and_logit(inp, att):
    out = model(input_ids=inp, attention_mask=att, output_hidden_states=True)
    last = att.sum(1) - 1  # right padding: last real token
    h = out.hidden_states[-1][torch.arange(len(inp), device=dev), last]
    return h, out.logits[:, 0].float()


model.train(); dom.train()
run_l = run_d = 0.0; acc_ok = acc_n = 0; t_step = time.time()
for s in range(steps):
    lam = args.max_lambda * (2.0 / (1.0 + math.exp(-10 * s / max(1, steps - 1))) - 1.0)
    sid = src_ids[s * args.bs:(s + 1) * args.bs]
    y = y_all[s * args.bs:(s + 1) * args.bs].to(dev)
    tid = [tgt_ids[j] for j in rng.integers(0, len(tgt_ids), len(sid))]
    mb = math.ceil(len(sid) / args.accum)
    ml = dl = 0.0
    for k in range(0, len(sid), mb):
        si, sa = collate(sid[k:k + mb])
        if args.max_lambda == 0:  # plain (self-)training: no target pass, no domain head (half the compute)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(input_ids=si, attention_mask=sa).logits[:, 0].float()
            m_loss = F.binary_cross_entropy_with_logits(logit, y[k:k + mb], reduction="sum") / len(sid)
            m_loss.backward(); ml += m_loss.item()
            continue
        ti, ta = collate(tid[k:k + mb])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs, logit = pooled_and_logit(si, sa)
            ht, _ = pooled_and_logit(ti, ta)
            d = dom(GradReverse.apply(torch.cat([hs, ht]).float(), lam))[:, 0]
        dy = torch.cat([torch.zeros(len(hs)), torch.ones(len(ht))]).to(dev)
        m_loss = F.binary_cross_entropy_with_logits(logit, y[k:k + mb], reduction="sum") / len(sid)
        d_loss = F.binary_cross_entropy_with_logits(d, dy, reduction="sum") / (2 * len(sid))
        (m_loss + d_loss).backward()
        ml += m_loss.item(); dl += d_loss.item()
        acc_ok += int(((d > 0).float() == dy).sum()); acc_n += len(dy)
    torch.nn.utils.clip_grad_norm_(params + list(dom.parameters()), 1.0)
    opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
    run_l += ml; run_d += dl
    if s == 1:
        torch.cuda.empty_cache()
    if (s + 1) % 200 == 0 or s == steps - 1:
        rate = 200 * args.bs / (time.time() - t_step); t_step = time.time()
        log(f"step {s + 1}/{steps} match loss {run_l / 200:.4f} | domain loss {run_d / 200:.4f} | domain acc "
            f"{acc_ok / max(acc_n, 1):.3f} | lambda {lam:.3f} | {rate:.0f} src pairs/s | "
            f"ETA {(steps - s - 1) * args.bs / max(rate, 1) / 60:.0f} min")
        run_l = run_d = 0.0; acc_ok = acc_n = 0
    if (s + 1) % args.save_every == 0 or s == steps - 1:
        model.save_pretrained(os.path.join(args.out, "adapter"))
tok.save_pretrained(os.path.join(args.out, "adapter"))
open(os.path.join(args.out, "adapter", "DONE"), "w").close()
log("DONE: score with qwen_score.py --adapter", os.path.join(args.out, "adapter"))
