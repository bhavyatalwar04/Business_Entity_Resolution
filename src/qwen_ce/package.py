"""Pack Qwen CE outputs in the CE handoff format expected by the laptop stacker.

~/work/Business_Entity_Resolution/handoff/ce_qwen_out/
  ce_oof_fold0_part*.parquet, ce_test_part*.parquet  (s1_id, pool_id, ce_prob float32; sorted by s1_id; zstd; < 90 MB each)
  fold0_metrics.json, ce_oof_fold0.DONE, ce_test.DONE
Usage: python package.py [oof|test|all]
"""
import json, os, sys

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
import paths as P

SRC = os.path.join(P.WORK, "out")
INP = P.INPUTS
DST = os.path.join(P.HANDOFF, "ce_qwen_out")
ROWS_PER_PART = 3_000_000  # ~25-30 MB per part at these column widths; checked below
os.makedirs(DST, exist_ok=True)
what = sys.argv[1] if len(sys.argv) > 1 else "all"


def parts(df, stem):
    df = df[["s1_id", "pool_id", "ce_prob"]].astype({"ce_prob": "float32"}).sort_values("s1_id", kind="stable")
    for i, s in enumerate(range(0, len(df), ROWS_PER_PART)):
        f = os.path.join(DST, f"{stem}_part{i:02d}.parquet")
        df.iloc[s:s + ROWS_PER_PART].to_parquet(f, compression="zstd", index=False)
        mb = os.path.getsize(f) / 2**20
        assert mb < 90, f"{f} is {mb:.1f} MB"
        print(f"{f}  {min(ROWS_PER_PART, len(df) - s):,} rows  {mb:.1f} MB")
    open(os.path.join(DST, f"{stem}.DONE"), "w").close()


if what in ("oof", "all"):
    oof = pd.read_parquet(os.path.join(SRC, "qwen_oof_fold0.parquet"))
    parts(oof, "ce_oof_fold0")
    f0 = pd.read_parquet(os.path.join(INP, "fold0.parquet")).merge(oof, on=["s1_id", "pool_id"])
    ll = lambda y, p: float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))
    h = f0[f0.in_handoff]
    metrics = {"model": "Qwen/Qwen3-4B-Base (Apache-2.0, 4.02B) seq-cls + LoRA r=32",
               "n_rows": int(len(f0)), "auc": float(roc_auc_score(f0.label, f0.ce_prob)), "logloss": ll(f0.label, f0.ce_prob),
               "handoff_rows": int(len(h)), "handoff_auc": float(roc_auc_score(h.label, h.ce_prob)),
               "handoff_logloss": ll(h.label, h.ce_prob)}
    json.dump(metrics, open(os.path.join(DST, "fold0_metrics.json"), "w"), indent=1)
    print(json.dumps(metrics, indent=1))

if what in ("test", "all"):
    parts(pd.read_parquet(os.path.join(SRC, "qwen_test_contested.parquet")), "ce_test")
