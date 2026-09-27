"""Offline test of the TransClean-style triangle filter on fold-0 (India/US labels), before laptop's real build.

Proxy decision (not the stacker): per S1, predict every candidate pair with v11 Qwen3 prob >= THR.
Filter: for each S1 whose predicted set has x in S2 and y in S3, if P_qwen(x, y) < t, drop the weaker of the two
(lower S1-side prob). Metric: macro F0.5 over S1 entities in fold0.parquet, truth = labelled positives in the same
table (an S1 with no positive is a singleton: empty prediction = 1.0, anything else = 0.0).
usage: python tri_sim.py
"""
import glob

import numpy as np
import pandas as pd

rd = lambda p: pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(p))], ignore_index=True)
f0 = pd.read_parquet("inputs/fold0.parquet")[["s1_id", "pool_id", "label"]]
q = pd.read_parquet("out/qwen_oof_fold0.parquet").rename(columns={"ce_prob": "p"})
d = f0.merge(q, on=["s1_id", "pool_id"])
tri = rd("out_tri/tri_fold0_part*.parquet").rename(columns={"s1_id": "s2_id", "pool_id": "s3_id", "ce_prob": "pxy"})
truth = d[d.label == 1].groupby("s1_id").pool_id.apply(set)
all_s1 = d.s1_id.unique()


def macro_f05(pred):
    s = 0.0
    for s1 in all_s1:
        t, p = truth.get(s1, set()), pred.get(s1, set())
        if not t:
            s += 1.0 if not p else 0.0
            continue
        tp = len(t & p)
        if tp == 0:
            continue
        pr, rc = tp / len(p), tp / len(t)
        s += 1.25 * pr * rc / (0.25 * pr + rc)
    return s / len(all_s1)


for THR in (0.5, 0.7):
    pr = d[d.p >= THR]
    base = pr.groupby("s1_id").pool_id.apply(set).to_dict()
    b = macro_f05(base)
    x = pr[pr.pool_id.str.startswith("S2")].rename(columns={"pool_id": "s2_id", "p": "px"})[["s1_id", "s2_id", "px"]]
    y = pr[pr.pool_id.str.startswith("S3")].rename(columns={"pool_id": "s3_id", "p": "py"})[["s1_id", "s3_id", "py"]]
    trip = x.merge(y, on="s1_id").merge(tri, on=["s2_id", "s3_id"], how="left")
    print(f"THR {THR}: base macro F0.5 {b:.5f} | predicted triangles {len(trip):,} (scored {trip.pxy.notna().mean():.1%})")
    for t in (0.01, 0.03, 0.1, 0.2, 0.3, 0.5):
        bad = trip[trip.pxy.notna() & (trip.pxy < t)]
        drop = set(zip(bad.s1_id, np.where(bad.px < bad.py, bad.s2_id, bad.s3_id)))
        pred = {s1: {p for p in ps if (s1, p) not in drop} for s1, ps in base.items()}
        print(f"   t={t:<5} drops {len(drop):>6,} matches -> macro F0.5 {macro_f05(pred):.5f} ({macro_f05(pred) - b:+.5f})")
