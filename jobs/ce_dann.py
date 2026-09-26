"""Domain-adversarial cross-encoder (DANN / gradient reversal, as in DADER, VLDB 2022) for an unseen country.

The matcher learns from labelled source pairs; a small domain classifier on the [CLS] vector tries to tell source pairs
from unlabelled target pairs, and a gradient-reversal layer pushes the encoder to make them indistinguishable, so the
source-trained matcher head transfers. No target labels are used or invented.

--mode proxy : source = INDIA labelled (full-data step-3 OOF, fold != 0), target = US fold-0 candidates (unlabelled,
               the analogue of France test). Two runs with identical data/seed: lam 0 (plain baseline) and lam > 0.
               Both score the real US + India fold-0 pairs (prob >= 0.001); B - baseline = value of the alignment.
Outputs (<out>): scores_lam<l>.parquet, dann_result.json

Usage: PYTHONPATH=. python jobs/ce_dann.py --mode proxy --out handoff/dann_proxy_out
"""
import argparse
import glob
import json
import math
import os
import time
import zlib

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.blend_eval import held_out
from src.ce_rescore import amp, attach_text, batches, load_texts, log, metrics, predict
from src.decide import FAST_GRID, decide, tune
from src.evaluate import f05
from src.io_utils import load_ground_truth, read_tsv


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def fold0(s):
    return zlib.crc32(s.encode()) % 5 == 0


def train_dann(args, tok, src_t1, src_t2, y, tgt_t1, tgt_t2, lam_max, ckpt):
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1).cuda()
    hid = model.config.hidden_size
    dom = torch.nn.Sequential(torch.nn.Linear(hid, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1)).cuda()
    params = list(model.parameters()) + list(dom.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, fused=True)
    order = rng.permutation(len(y))
    total = (len(order) + args.bs - 1) // args.bs
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * total), total)
    t_order = rng.permutation(len(tgt_t1))
    t_bs = args.bs // 2
    tgt_iter = batches(tgt_t1, tgt_t2, np.resize(t_order, total * t_bs), t_bs, tok, args.max_len)
    bce = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, step, run = time.time(), 0, [0.0, 0.0, 0]
    log(f"DANN lam_max {lam_max}: {len(y):,} source pairs, {len(tgt_t1):,} target pairs, {total:,} steps")
    for _, b in batches(src_t1, src_t2, order, args.bs, tok, args.max_len, labels=y):
        _, tb = next(tgt_iter)
        b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
        tb = {k: v.cuda(non_blocking=True) for k, v in tb.items()}
        lab = b.pop("labels")
        p = step / total
        lam = lam_max * (2 / (1 + math.exp(-10 * p)) - 1)
        with amp():
            out = model(**b, output_hidden_states=True)
            loss_m = bce(out.logits.squeeze(-1).float(), lab)
            loss = loss_m
            if lam_max > 0:
                h_s = out.hidden_states[-1][:, 0]
                h_t = model(**tb, output_hidden_states=True).hidden_states[-1][:, 0]
                h = GradReverse.apply(torch.cat([h_s, h_t]), lam)
                d = dom(h).squeeze(-1).float()
                dl = torch.cat([torch.zeros(len(h_s)), torch.ones(len(h_t))]).cuda()
                loss_d = bce(d, dl)
                loss = loss_m + loss_d
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 50 == 0:
            run[0] += loss_m.item(); run[1] += (loss_d.item() if lam_max > 0 else 0.0); run[2] += 1
        if step % 1000 == 0:
            el = time.time() - t0
            log(f"step {step:,}/{total:,} match loss {run[0] / max(run[2], 1):.4f} domain loss {run[1] / max(run[2], 1):.4f} "
                f"lam {lam:.3f} {el / 60:.1f} min, eta {el / step * (total - step) / 60:.0f} min")
            run = [0.0, 0.0, 0]
    final = os.path.join(ckpt, "final")
    model.save_pretrained(final)
    tok.save_pretrained(final)
    log(f"saved {final}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="proxy", choices=["proxy"])
    ap.add_argument("--base", default="FacebookAI/xlm-roberta-large")
    ap.add_argument("--out", default="handoff/dann_proxy_out")
    ap.add_argument("--ckpt", default="artefacts/dann_proxy")
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.1])
    ap.add_argument("--cap_src", type=int, default=1_000_000)
    ap.add_argument("--cap_tgt", type=int, default=1_000_000)
    ap.add_argument("--rest_frac", type=float, default=0.03)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--infer_bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    res = {"lams": args.lams}

    gold = load_ground_truth("dataset")
    s1 = read_tsv("dataset/train/train_source1.tsv")
    country = dict(zip(s1["entity_id"], s1["country"].str.lower()))
    texts = load_texts("dataset", "train")
    d = ds.dataset(sorted(glob.glob("handoff/full_out/oof_train_full_part*.parquet")), format="parquet")
    oof = d.to_table(filter=ds.field("prob") >= 0.001).to_pandas()
    oof["country"] = oof["s1_id"].map(country)
    oof["label"] = np.fromiter((p in gold.get(s, ()) for s, p in zip(oof["s1_id"], oof["pool_id"])), bool, len(oof))

    src = oof[(oof["fold"] != 0) & (oof["country"] == "india")][["s1_id", "pool_id", "label"]]
    t = d.to_table(filter=(ds.field("prob") < 0.001) & (ds.field("fold") != 0), columns=["s1_id", "pool_id"])
    t = t.take(np.sort(rng.choice(t.num_rows, int(t.num_rows * args.rest_frac), replace=False))).to_pandas()
    t = t[t["s1_id"].map(country) == "india"]
    t["label"] = np.fromiter((p in gold.get(s, ()) for s, p in zip(t["s1_id"], t["pool_id"])), bool, len(t))
    src = pd.concat([src, t], ignore_index=True)
    src = src.sample(min(len(src), args.cap_src), random_state=args.seed).reset_index(drop=True)
    tgt = oof[(oof["fold"] == 0) & (oof["country"] == "us")][["s1_id", "pool_id"]]
    tgt = tgt.sample(min(len(tgt), args.cap_tgt), random_state=args.seed).reset_index(drop=True)
    ev = oof[(oof["fold"] == 0) & oof["country"].isin(["us", "india"])][["s1_id", "pool_id", "label", "country"]]
    ev = ev.reset_index(drop=True)
    res["data"] = {"src": int(len(src)), "src_pos": int(src["label"].sum()), "tgt_unlabelled": int(len(tgt)), "eval": int(len(ev))}
    log(f"data {res['data']}")
    if args.dry:
        return

    s1t, s2t = attach_text(src, texts)
    tg1, tg2 = attach_text(tgt, texts)
    e1, e2 = attach_text(ev, texts)
    y = src["label"].to_numpy().astype(np.float32)
    tok = AutoTokenizer.from_pretrained(args.base)

    s1_0 = s1[s1["entity_id"].map(fold0)]
    ids = {c: sorted(s1_0.loc[s1_0["country"].str.lower() == c, "entity_id"]) for c in ("us", "india")}
    halves = [[s for s in ids["us"] if zlib.crc32(s.encode()) % 10 == 0], [s for s in ids["us"] if zlib.crc32(s.encode()) % 10 == 5]]
    for lam in args.lams:
        name = f"lam{lam:g}"
        f = os.path.join(args.out, f"scores_{name}.parquet")
        if os.path.exists(f):
            p = pd.read_parquet(f)["ce_prob"].to_numpy()
        else:
            model = train_dann(args, tok, s1t, s2t, y, tg1, tg2, lam, os.path.join(args.ckpt, name))
            p = predict(model, tok, e1, e2, args)
            pd.DataFrame({"s1_id": ev["s1_id"], "pool_id": ev["pool_id"], "ce_prob": p}).to_parquet(f)
            del model
            torch.cuda.empty_cache()
        ev[name] = p
        r = {}
        for c in ("us", "india"):
            m = ev["country"] == c
            r[c] = metrics(ev.loc[m, "label"].to_numpy(), ev.loc[m, name].to_numpy())
        ho, _, _ = held_out(ev, name, gold, halves, {s: "us" for s in ids["us"]})
        dsub = ev[["s1_id", "pool_id", name]].rename(columns={name: "prob"})
        params, _ = tune(dsub[dsub["s1_id"].isin(set(ids["india"]))], gold, ids["india"], verbose=False, grid=FAST_GRID, o2o_opts=(True,))
        pred = decide(dsub[dsub["s1_id"].isin(set(ids["us"]))], ids["us"], params)
        tf = float(np.mean([f05(pred[s], gold.get(s, ())) for s in ids["us"]]))
        r.update(us_heldout_f05=ho, us_transfer_f05=tf, us_mean_prob_nonmatch=float(ev.loc[(ev["country"] == "us") & ~ev["label"], name].mean()))
        res[name] = r
        log(f"{name}: US AUC {r['us']['auc']:.5f} logloss {r['us']['logloss']:.4f} | India AUC {r['india']['auc']:.5f} | "
            f"US held-out F0.5 {ho:.4f} | transfer {tf:.4f} | US mean p(non-match) {r['us_mean_prob_nonmatch']:.3f}")
        json.dump(res, open(os.path.join(args.out, "dann_result.json"), "w"), indent=1, default=str)
    log("done")


if __name__ == "__main__":
    main()
