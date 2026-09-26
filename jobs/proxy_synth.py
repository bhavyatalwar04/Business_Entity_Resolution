"""Leave-one-country-out proxy for the synthetic-pair method (src/synth_pairs.py). US plays the unseen country.

A : xlm-roberta-large fine-tuned from the public checkpoint on INDIA-only labelled pairs (full-data step-3 OOF, fold != 0).
B : A continued on synthetic US pairs (generator operators applied to US fold-0 S1 texts, the same thing we would do to
    France test S1) mixed 1:1 with India replay pairs.
Both score the REAL US and India fold-0 pairs (step-3 prob >= 0.001). The measured value of the method is B - A on US:
  - held-out : decision tuned on one half of the US fold-0 S1, scored on the other, both ways (ranking quality)
  - transfer : decision tuned on India fold-0, applied to US (what we can do for France)
Reference: ce_full (trained WITH US labels) on the same rows, i.e. the ceiling the proxy gap is measured against.

Outputs (<out>): scores_A.parquet, scores_B.parquet, synth_us.parquet, proxy_result.json
Usage: PYTHONPATH=. python jobs/proxy_synth.py --out handoff/proxy_synth_out
"""
import argparse
import glob
import json
import os
import zlib

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from transformers import AutoTokenizer

from src.blend_eval import held_out
from src.ce_rescore import attach_text, load_texts, log, metrics, predict, read_pairs, train
from src.decide import FAST_GRID, decide, tune
from src.evaluate import f05
from src.io_utils import load_ground_truth, read_tsv
from src.synth_pairs import build_pairs


def load_oof(min_prob):
    d = ds.dataset(sorted(glob.glob("handoff/full_out/oof_train_full_part*.parquet")), format="parquet")
    return d.to_table(filter=ds.field("prob") >= min_prob).to_pandas()


def label(df, gold):
    return np.fromiter((p in gold.get(s, ()) for s, p in zip(df["s1_id"], df["pool_id"])), bool, len(df)).astype(np.int8)


def transfer(df, col, gold, src_ids, dst_ids):
    d = df[["s1_id", "pool_id", col]].rename(columns={col: "prob"})
    params, _ = tune(d[d["s1_id"].isin(set(src_ids))], gold, src_ids, verbose=False, grid=FAST_GRID, o2o_opts=(True,))
    pred = decide(d[d["s1_id"].isin(set(dst_ids))], dst_ids, params)
    return float(np.mean([f05(pred[s], gold.get(s, ())) for s in dst_ids])), params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="FacebookAI/xlm-roberta-large")
    ap.add_argument("--out", default="handoff/proxy_synth_out")
    ap.add_argument("--ckpt", default="artefacts/proxy_synth")
    ap.add_argument("--cap_a", type=int, default=1_500_000)
    ap.add_argument("--rest_frac", type=float, default=0.03)
    ap.add_argument("--n_pos", type=int, default=2)
    ap.add_argument("--cap_anchors", type=int, default=100_000)
    ap.add_argument("--n_dec", type=int, default=2)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--infer_bs", type=int, default=1024)
    ap.add_argument("--lr_a", type=float, default=1e-5)
    ap.add_argument("--lr_b", type=float, default=5e-6)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry", action="store_true", help="data prep + synthetic pairs only, no GPU")
    args = ap.parse_args()
    args.epochs = 1
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    res = {}

    gold = load_ground_truth("dataset")
    s1 = read_tsv("dataset/train/train_source1.tsv")
    country = dict(zip(s1["entity_id"], s1["country"].str.lower()))
    texts = load_texts("dataset", "train")

    oof = load_oof(0.001)
    oof["country"] = oof["s1_id"].map(country)
    oof["label"] = label(oof, gold)
    log(f"step-3 OOF rows prob>=0.001: {len(oof):,}")

    # --- A: India-only training pairs (fold != 0)
    tr = oof[(oof["fold"] != 0) & (oof["country"] == "india")]
    d = ds.dataset(sorted(glob.glob("handoff/full_out/oof_train_full_part*.parquet")), format="parquet")
    t = d.to_table(filter=(ds.field("prob") < 0.001) & (ds.field("fold") != 0), columns=["s1_id", "pool_id"])
    t = t.take(np.sort(rng.choice(t.num_rows, int(t.num_rows * args.rest_frac), replace=False)))
    rest = t.to_pandas()
    del t
    rest = rest[rest["s1_id"].map(country) == "india"]
    rest["label"] = label(rest, gold)
    tr = pd.concat([tr[["s1_id", "pool_id", "label"]], rest[["s1_id", "pool_id", "label"]]], ignore_index=True)
    del rest
    if len(tr) > args.cap_a:
        tr = tr.sample(args.cap_a, random_state=args.seed).reset_index(drop=True)
    res["train_a"] = {"pairs": int(len(tr)), "pos": int(tr["label"].sum())}

    if args.dry:
        us0 = s1[(s1["country"].str.lower() == "us") & s1["entity_id"].map(lambda s: zlib.crc32(s.encode()) % 5 == 0)]
        syn = build_pairs(us0[["entity_id", "business_name", "business_address"]].head(20000), "US", rng, args.n_pos, args.n_dec)
        ev = oof[(oof["fold"] == 0) & oof["country"].isin(["us", "india"])]
        log(f"DRY train A {res['train_a']} | eval rows {len(ev):,} | synth sample {len(syn):,} {syn['kind'].value_counts().to_dict()}")
        return
    tok = AutoTokenizer.from_pretrained(args.base)
    a_dir = os.path.join(args.ckpt, "A")
    os.makedirs(a_dir, exist_ok=True)
    if os.path.exists(os.path.join(a_dir, "final", "config.json")):
        log("A already trained")
    else:
        args.model, args.lr = args.base, args.lr_a
        args.ckpt = a_dir
        train(args, tok, texts, tr)
        args.ckpt = os.path.dirname(a_dir)

    # --- eval rows: real fold-0 pairs of US and India
    ev = oof[(oof["fold"] == 0) & oof["country"].isin(["us", "india"])][["s1_id", "pool_id", "label", "country"]]
    ev = ev.reset_index(drop=True)
    e1, e2 = attach_text(ev, texts)
    log(f"eval rows {len(ev):,} (US {int((ev['country'] == 'us').sum()):,})")

    from transformers import AutoModelForSequenceClassification
    import torch

    def score(path, name):
        f = os.path.join(args.out, f"scores_{name}.parquet")
        if os.path.exists(f):
            return pd.read_parquet(f)["ce_prob"].to_numpy()
        model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1).cuda()
        p = predict(model, tok, e1, e2, args)
        pd.DataFrame({"s1_id": ev["s1_id"], "pool_id": ev["pool_id"], "ce_prob": p}).to_parquet(f)
        del model
        torch.cuda.empty_cache()
        return p

    ev["A"] = score(os.path.join(a_dir, "final"), "A")

    # --- B: synthetic US from US fold-0 S1 texts + India replay
    us0 = s1[(s1["country"].str.lower() == "us") & s1["entity_id"].map(lambda s: zlib.crc32(s.encode()) % 5 == 0)]
    us0 = us0.sample(min(len(us0), args.cap_anchors), random_state=args.seed)
    syn = build_pairs(us0[["entity_id", "business_name", "business_address"]], "US", rng, args.n_pos, args.n_dec)
    syn.to_parquet(os.path.join(args.out, "synth_us.parquet"))
    res["synth"] = {"anchors": int(len(us0)), "pairs": int(len(syn)), "pos": int(syn["label"].sum()),
                    "kinds": syn["kind"].value_counts().to_dict()}
    log(f"synthetic US pairs {len(syn):,} from {len(us0):,} anchors: {res['synth']['kinds']}")
    texts_b = pd.concat([texts, pd.Series(syn["pool_text"].values, index=syn["pool_id"].values)])
    replay = tr.sample(min(len(tr), len(syn)), random_state=args.seed + 1)
    tr_b = pd.concat([syn[["s1_id", "pool_id", "label"]], replay], ignore_index=True)
    b_dir = os.path.join(args.ckpt, "B")
    os.makedirs(b_dir, exist_ok=True)
    if not os.path.exists(os.path.join(b_dir, "final", "config.json")):
        args.model, args.lr, args.ckpt = os.path.join(a_dir, "final"), args.lr_b, b_dir
        train(args, tok, texts_b, tr_b)
    ev["B"] = score(os.path.join(b_dir, "final"), "B")

    # --- C (control): A continued on the same number of India-only pairs, so B - C isolates the synthetic data
    tr_c = tr.drop(replay.index).sample(min(len(tr) - len(replay), len(syn)), random_state=args.seed + 2)
    tr_c = pd.concat([tr_c, replay], ignore_index=True)
    c_dir = os.path.join(os.path.dirname(b_dir), "C")
    os.makedirs(c_dir, exist_ok=True)
    if not os.path.exists(os.path.join(c_dir, "final", "config.json")):
        args.model, args.lr, args.ckpt = os.path.join(a_dir, "final"), args.lr_b, c_dir
        train(args, tok, texts, tr_c)
    ev["C"] = score(os.path.join(c_dir, "final"), "C")

    # --- reference: ce_full trained with US labels (covers the same step-3 fold-0 rows)
    ref = glob.glob("handoff/ce_full_out/ce_oof_fold0_part*.parquet") + glob.glob("handoff/ce_full_out/ce_extra_fold0_part*.parquet")
    cols = ["A", "C", "B"]
    if ref:
        r = pd.concat([pd.read_parquet(f, columns=["s1_id", "pool_id", "ce_prob"]) for f in ref]).drop_duplicates(["s1_id", "pool_id"])
        ev = ev.merge(r.rename(columns={"ce_prob": "ce_full_ref"}), on=["s1_id", "pool_id"], how="left")
        res["ref_coverage"] = float(ev["ce_full_ref"].notna().mean())
        ev["ce_full_ref"] = ev["ce_full_ref"].fillna(0.0)
        cols.append("ce_full_ref")

    # --- evaluation on ALL fold-0 S1 of each country (singletons included)
    s1_0 = s1[s1["entity_id"].map(lambda s: zlib.crc32(s.encode()) % 5 == 0)]
    ids = {c: sorted(s1_0.loc[s1_0["country"].str.lower() == c, "entity_id"]) for c in ("us", "india")}
    halves = [[s for s in ids["us"] if zlib.crc32(s.encode()) % 10 == 0], [s for s in ids["us"] if zlib.crc32(s.encode()) % 10 == 5]]
    for c in cols:
        m = metrics(ev.loc[ev["country"] == "us", "label"].to_numpy(), ev.loc[ev["country"] == "us", c].to_numpy())
        ho, _, _ = held_out(ev, c, gold, halves, {s: "us" for s in ids["us"]})
        tf, params = transfer(ev, c, gold, ids["india"], ids["us"])
        res[c] = {"us_pair": m, "us_heldout_f05": ho, "us_transfer_f05": tf, "transfer_params": str(params)}
        log(f"{c:12s} US pair AUC {m['auc']:.5f} logloss {m['logloss']:.5f} | US held-out F0.5 {ho:.4f} | "
            f"transfer (tuned on India) {tf:.4f}")
    json.dump(res, open(os.path.join(args.out, "proxy_result.json"), "w"), indent=1, default=str)
    log("done")


if __name__ == "__main__":
    main()
