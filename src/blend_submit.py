"""Blend LightGBM (v2) and cross-encoder probabilities, tune the decision on fold 0, write test submissions.

Fold 0 = held-out S1s of the 500k training sample (never seen by either model). Each blend weight is scored
honestly: decision params tuned on one half of fold 0 and scored on the other half, both ways round.
Test variants shift probabilities of S1s whose country never occurs in training (logit - delta):
  base   : delta 0      strict : delta +DELTA      loose : delta -DELTA
Countries are compared as open-set string labels; nothing is hard-coded.

Usage: python -m src.blend_submit [--delta 1.0]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src.decide import decide, tune
from src.evaluate import f05
from src.io_utils import load_ground_truth, read_tsv, write_tsv

THR = 0.001
WEIGHTS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.85, 1.0]
EPS = 1e-6


def logit(p):
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def blend(lgbm, ce, w):
    return (1 / (1 + np.exp(-(w * logit(ce) + (1 - w) * logit(lgbm))))).astype(np.float32)


def read_parts(pattern, columns=None):
    return pd.concat([pd.read_parquet(f, columns=columns) for f in sorted(glob.glob(pattern))], ignore_index=True)


def score(df, gold, ids, params):
    pred = decide(df, ids, params)
    return float(np.mean([f05(pred[i], gold.get(i, ())) for i in ids]))


G = {}


def eval_weight(w):
    df, gold, halves = G["df"], G["gold"], G["halves"]
    d = df.assign(prob=blend(df["lgbm_prob"].values, df["ce_prob"].values, w))[["s1_id", "pool_id", "prob"]]
    honest = []
    for a, b in ((0, 1), (1, 0)):
        pa, _ = tune(d[d["s1_id"].isin(halves[a])], gold, halves[a], verbose=False)
        honest.append((score(d[d["s1_id"].isin(halves[b])], gold, halves[b], pa), len(halves[b])))
    h = sum(s * n for s, n in honest) / sum(n for _, n in honest)
    params, full = tune(d, gold, halves[0] + halves[1], verbose=False)
    print(f"w_ce {w:.2f}: honest macro F0.5 {h:.5f} | tuned-on-all {full:.5f} | {params}", flush=True)
    return w, h, full, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--out", default="output/blend")
    args = ap.parse_args()

    tr = read_parts("handoff/ce/train_pairs_part*.parquet")
    f0 = tr[tr["fold"] == 0]
    s1_f0 = sorted(f0["s1_id"].unique())
    ce0 = read_parts("handoff/ce_out/ce_oof_fold0_part*.parquet")
    df = f0[f0["lgbm_prob"] >= THR].merge(ce0, on=["s1_id", "pool_id"], how="left")
    assert df["ce_prob"].notna().all()
    del tr, f0
    gold_all = load_ground_truth("dataset")
    gold = {s: gold_all.get(s, set()) for s in s1_f0}
    halves = [[], []]
    for s in s1_f0:
        halves[zlib.crc32(s.encode()) & 1].append(s)
    print(f"fold-0: {len(s1_f0):,} S1, {len(df):,} pairs with lgbm_prob >= {THR}", flush=True)
    G.update(df=df, gold=gold, halves=halves)
    with Pool(len(WEIGHTS)) as pool:
        res = pool.map(eval_weight, WEIGHTS)
    best_w, best_h, best_full, params = max(res, key=lambda r: r[1])
    print(f"BEST w_ce {best_w} honest {best_h:.5f} (lgbm only: {res[0][1]:.5f}, ce only: {res[-1][1]:.5f})", flush=True)

    te = read_parts("handoff/ce/test_pairs_part*.parquet").merge(
        read_parts("handoff/ce_out/ce_test_part*.parquet"), on=["s1_id", "pool_id"], how="left")
    assert te["ce_prob"].notna().all()
    te["prob"] = blend(te["lgbm_prob"].values, te["ce_prob"].values, best_w)
    s1 = read_tsv("dataset/test/test_source1.tsv", usecols=["entity_id", "country"])
    train_countries = set(read_tsv("dataset/train/train_source1.tsv", usecols=["country"])["country"].unique())
    unseen_s1 = set(s1.loc[~s1["country"].isin(train_countries), "entity_id"])
    country = dict(zip(s1["entity_id"], s1["country"]))
    s1_ids = s1["entity_id"].tolist()
    print(f"test: {len(s1_ids):,} S1, unseen-country S1 {len(unseen_s1):,} "
          f"({sorted(set(s1['country']) - train_countries)})", flush=True)
    unseen_mask = te["s1_id"].isin(unseen_s1).values
    summary = {"w_ce": best_w, "honest_f05": best_h, "tuned_on_all_f05": best_full, "params": params,
               "all_weights": [{"w_ce": w, "honest": h, "full": f} for w, h, f, _ in res], "variants": {}}
    for name, delta in (("base", 0.0), ("strict", args.delta), ("loose", -args.delta)):
        d = te[["s1_id", "pool_id"]].copy()
        z = logit(te["prob"].values)
        z[unseen_mask] -= delta
        d["prob"] = (1 / (1 + np.exp(-z))).astype(np.float32)
        matches = decide(d, s1_ids, params)
        out = os.path.join(args.out, name)
        path = os.path.join(out, "matching_results.tsv")
        write_tsv(s1_ids, matches, path, "matched_entity_ids")
        stats = {}
        for c in sorted(set(country.values())):
            ids = [s for s in s1_ids if country[s] == c]
            stats[c] = {"empty": round(sum(not matches[s] for s in ids) / len(ids), 4),
                        "per_s1": round(sum(len(matches[s]) for s in ids) / len(ids), 3)}
        summary["variants"][name] = {"delta_unseen": delta, "per_country": stats}
        print(f"[{name}] delta {delta:+.1f} -> {stats}", flush=True)
        r = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", path,
                            "--test-dir", "dataset/test"], capture_output=True, text=True)
        print((r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else "", flush=True)
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
