"""CE base retrained on the full step-3 train candidates (all 2.2M S1), fold != 0 only.

Train : fold_of(s1_id, 5) != 0 rows of artefacts/train/oof.parquet; all positives + negatives with step-3 prob >= 0.001
        + 5% of the other negatives; capped at --cap pairs (random fill is cut first).
Score : (a) fold-0 pairs = step-3 OOF fold 0 with prob >= 0.001  UNION  handoff fold-0 pairs with lgbm_prob >= 0.001
        (b) test pairs   = probs_model_full with prob >= 0.001    UNION  handoff test pairs
Output: <out>/ce_oof_fold0_part*, ce_test_part* (s1_id, pool_id, ce_prob), fold0_metrics.json, *.DONE markers.

Usage: PYTHONPATH=. python -m src.xlmr_ce.ce_full --ckpt artefacts/ce_full --out handoff/ce_full_out
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.xlmr_ce.ce_rescore import attach_text, load_texts, log, metrics, predict, read_pairs, train, write_parts
from src.evaluate import fold_of
from src.io_utils import read_tsv

THR = 0.001


def lgt(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ids(split):
    d = f"artefacts/{split}"
    s1 = pd.read_parquet(f"{d}/s1.parquet", columns=["entity_id"])["entity_id"].to_numpy()
    pool = np.concatenate([pd.read_parquet(f"{d}/s{k}.parquet", columns=["entity_id"])["entity_id"].to_numpy() for k in (2, 3)])
    return s1, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--ckpt", default="artefacts/ce_full")
    ap.add_argument("--out", default="handoff/ce_full_out")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--infer_bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--rest_frac", type=float, default=0.05)
    ap.add_argument("--cap", type=int, default=8_000_000)
    ap.add_argument("--save_every", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_prev", action="store_true",
                    help="epoch 2: rebuild the first run's sample (seed 42, cap 4M, rest 5%%) and train only on rows NOT in it")
    ap.add_argument("--prev_cap", type=int, default=4_000_000)
    ap.add_argument("--contested", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="score only pairs whose 0.5/0.5 logit blend of step-3 LightGBM and ce_full lies in [LO, HI] "
                         "(fold 0 keeps ALL handoff fold-0 pairs for an honest blend_eval)")
    args = ap.parse_args()
    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- step-3 OOF with labels and folds (integer ids -> strings only where needed)
    s1, pool = ids("train")
    oof = pd.read_parquet("artefacts/train/oof.parquet", columns=["s1_id", "pool_id", "prob"])
    gt = read_tsv("dataset/train/train_ground_truth.tsv")
    g = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
    g = g[g["m"].fillna("") != ""]
    ga = pd.Index(s1).get_indexer(g["source1_entity_id"].to_numpy())
    gb = pd.Index(pool).get_indexer(g["m"].to_numpy())
    ok = (ga >= 0) & (gb >= 0)
    n_pool = len(pool)
    key = oof["s1_id"].to_numpy(np.int64) * n_pool + oof["pool_id"].to_numpy(np.int64)
    oof["label"] = np.isin(key, ga[ok].astype(np.int64) * n_pool + gb[ok]).astype(np.int8)
    del key, g, gt
    fold_s1 = np.fromiter((fold_of(x, 5) for x in s1), np.int8, len(s1))
    oof["fold"] = fold_s1[oof["s1_id"].to_numpy()]
    log(f"step-3 OOF {len(oof):,} rows, {int(oof['label'].sum()):,} positives, "
        f"fold-0 S1 {int((fold_s1 == 0).sum()):,}")

    # --- training sample (fold != 0)
    tr = oof[oof["fold"] != 0]
    core = (tr["label"].to_numpy() == 1) | (tr["prob"].to_numpy() >= THR)
    if args.exclude_prev:
        # replay the first run's selection exactly (same rng sequence as its code path), then drop those rows
        prng = np.random.RandomState(42)
        pfill = ~core & (prng.rand(len(tr)) < 0.05)
        pc, pf = np.flatnonzero(core), np.flatnonzero(pfill)
        prev = prng.choice(pc, args.prev_cap, replace=False) if len(pc) >= args.prev_cap else np.concatenate(
            [pc, pf if len(pc) + len(pf) <= args.prev_cap else prng.choice(pf, args.prev_cap - len(pc), replace=False)])
        unseen = np.ones(len(tr), bool)
        unseen[prev] = False
        log(f"epoch 2: first run used {len(prev):,} rows; {int(unseen.sum()):,} rows unseen")
        core = core & unseen
        fill = ~(tr["label"].to_numpy() == 1) & ~(tr["prob"].to_numpy() >= THR) & unseen & (rng.rand(len(tr)) < args.rest_frac)
    else:
        fill = ~core & (rng.rand(len(tr)) < args.rest_frac)
    n_core, n_fill = int(core.sum()), int(fill.sum())
    if n_core >= args.cap:
        keep = np.flatnonzero(core)
        keep = rng.choice(keep, args.cap, replace=False)
    else:
        f_idx = np.flatnonzero(fill)
        if n_core + len(f_idx) > args.cap:
            f_idx = rng.choice(f_idx, args.cap - n_core, replace=False)
        keep = np.concatenate([np.flatnonzero(core), f_idx])
    tr = tr.iloc[np.sort(keep)]
    tr = pd.DataFrame({"s1_id": s1[tr["s1_id"].to_numpy()], "pool_id": pool[tr["pool_id"].to_numpy()],
                       "label": tr["label"].to_numpy()})
    log(f"train sample: core {n_core:,} (pos + prob>={THR}), random fill {n_fill:,} -> using {len(tr):,}")

    # --- fold-0 scoring list
    f0 = oof[(oof["fold"] == 0) & (oof["prob"] >= THR)]
    f0 = pd.DataFrame({"s1_id": s1[f0["s1_id"].to_numpy()], "pool_id": pool[f0["pool_id"].to_numpy()],
                       "prob": f0["prob"].to_numpy(), "label": f0["label"].to_numpy()})
    del oof
    if args.contested:
        cf = pd.read_parquet("handoff/ce_full_out/ce_oof_fold0_part00.parquet")
        f0 = f0.merge(cf, on=["s1_id", "pool_id"], how="left").fillna({"ce_prob": 0.0})
        bl = 1 / (1 + np.exp(-(0.5 * lgt(f0["prob"].to_numpy()) + 0.5 * lgt(f0["ce_prob"].to_numpy()))))
        f0 = f0[(bl >= args.contested[0]) & (bl <= args.contested[1])].drop(columns="ce_prob")
        log(f"contested fold-0 step-3 pairs: {len(f0):,}")
    h0 = read_pairs("handoff/ce/train_pairs_part*.parquet")
    h0 = h0[(h0["fold"] == 0) & (h0["lgbm_prob"] >= THR)][["s1_id", "pool_id", "label"]]
    fold0 = pd.concat([f0[["s1_id", "pool_id", "label"]], h0], ignore_index=True).drop_duplicates(["s1_id", "pool_id"])
    log(f"fold-0 list: step-3 {len(f0):,} + handoff {len(h0):,} -> union {len(fold0):,}")

    tok = AutoTokenizer.from_pretrained(args.model)
    texts = load_texts("dataset", "train")
    final = os.path.join(args.ckpt, "final")
    t0 = time.time()
    if os.path.exists(os.path.join(final, "config.json")):
        model = AutoModelForSequenceClassification.from_pretrained(final).cuda()
    else:
        model = train(args, tok, texts, tr=tr)
    del tr
    t1, t2 = attach_text(fold0, texts)
    fold0["ce_prob"] = predict(model, tok, t1, t2, args)
    del texts, t1, t2
    m = fold0.merge(f0[["s1_id", "pool_id", "prob"]], on=["s1_id", "pool_id"])
    res = {"n_union": int(len(fold0)), "n_step3_rows": int(len(m)), "pos_step3_rows": int(m["label"].sum()),
           "ce_full_on_step3_rows": metrics(m["label"].values, m["ce_prob"].values),
           "step3_lgbm_on_step3_rows": metrics(m["label"].values, m["prob"].values),
           "minutes_so_far": round((time.time() - t0) / 60, 1)}
    json.dump(res, open(os.path.join(args.out, "fold0_metrics.json"), "w"), indent=1)
    log(f"fold-0 metrics {res}")
    write_parts(fold0[["s1_id", "pool_id", "ce_prob"]], args.out, "ce_oof_fold0")
    open(os.path.join(args.out, "ce_oof_fold0.DONE"), "w").close()

    # --- test (step 3 may still be predicting test when training ends: wait for rank:test to finish)
    while not (os.path.exists("artefacts/test/probs_model_full.parquet")
               and "--stage decide --split test" in open("logs/full.log").read().split("restart from rank train")[-1]):
        log("waiting for step-3 rank:test (probs_model_full) ...")
        time.sleep(120)
    s1t, poolt = ids("test")
    te = pd.read_parquet("artefacts/test/probs_model_full.parquet", columns=["s1_id", "pool_id", "prob"])
    te = te[te["prob"] >= THR]
    if pd.api.types.is_integer_dtype(te["s1_id"]):
        te = te.assign(s1_id=s1t[te["s1_id"].to_numpy()], pool_id=poolt[te["pool_id"].to_numpy()])
    ht = read_pairs("handoff/ce/test_pairs_part*.parquet", columns=["s1_id", "pool_id"])
    test = pd.concat([te[["s1_id", "pool_id"]], ht], ignore_index=True).drop_duplicates()
    if args.contested:
        cft = pd.concat([pd.read_parquet(f) for f in glob.glob("handoff/ce_full_out/ce_test_part*.parquet")])
        test = test.merge(te[["s1_id", "pool_id", "prob"]], on=["s1_id", "pool_id"], how="left").fillna({"prob": 0.0005})
        test = test.merge(cft, on=["s1_id", "pool_id"], how="left").fillna({"ce_prob": 0.0})
        bl = 1 / (1 + np.exp(-(0.5 * lgt(test["prob"].to_numpy()) + 0.5 * lgt(test["ce_prob"].to_numpy()))))
        test = test[(bl >= args.contested[0]) & (bl <= args.contested[1])][["s1_id", "pool_id"]].reset_index(drop=True)
        log(f"contested test pairs: {len(test):,}")
    log(f"test list: step-3 {len(te):,} + handoff {len(ht):,} -> union {len(test):,}")
    del te, ht
    texts = load_texts("dataset", "test")
    t1, t2 = attach_text(test, texts)
    del texts
    test["ce_prob"] = predict(model, tok, t1, t2, args)
    write_parts(test, args.out, "ce_test")
    open(os.path.join(args.out, "ce_test.DONE"), "w").close()
    json.dump(json.load(open(os.path.join(args.ckpt, "train_info.json"))) if os.path.exists(os.path.join(args.ckpt, "train_info.json")) else {},
              open(os.path.join(args.out, "train_info.json"), "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
