"""Contested-subset comparison on fold 0 (hpc #189): AUC / logloss of Qwen, XL, ce_full, ce_large and LightGBM on the
SAME fold-0 pairs, overall, on the handoff rows, and on the contested set
(sigmoid(0.5 logit(lgbm_full) + 0.5 logit(ce_full)) in [0.02, 0.98], logit clip 1e-6; XL's reference 0.873 / lgbm 0.887).
"""
import glob, os

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

I = os.path.expanduser("~/work/qwen_ce/inputs")
Q = os.path.expanduser("~/work/qwen_ce/out/qwen_oof_fold0.parquet")
rd = lambda p: pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(I, p)))], ignore_index=True)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

f0 = pd.read_parquet(f"{I}/fold0.parquet")[["s1_id", "pool_id", "label", "in_handoff"]]
df = f0.merge(pd.read_parquet(Q).rename(columns={"ce_prob": "qwen"}), on=["s1_id", "pool_id"])
df = df.merge(rd("ce_full/ce_oof_fold0_part*.parquet").rename(columns={"ce_prob": "ce_full"}), on=["s1_id", "pool_id"], how="left")
df = df.merge(rd("xl/ce_oof_fold0_part*.parquet").rename(columns={"ce_prob": "xl"}), on=["s1_id", "pool_id"], how="left")
df = df.merge(rd("ce_oof_fold0_part00.parquet").rename(columns={"ce_prob": "ce_large"}), on=["s1_id", "pool_id"], how="left")
oof = rd("full/oof_train_full_part*.parquet")
df = df.merge(oof[oof.fold == 0][["s1_id", "pool_id", "prob"]].rename(columns={"prob": "lgbm_full"}), on=["s1_id", "pool_id"], how="left")
z = 1 / (1 + np.exp(-(0.5 * lg(df.lgbm_full.fillna(0.0)) + 0.5 * lg(df.ce_full.fillna(0.0)))))
df["contested"] = (z >= 0.02) & (z <= 0.98) & df.lgbm_full.notna() & df.ce_full.notna()
print(f"fold-0 rows with Qwen {len(df):,} | handoff {int(df.in_handoff.sum()):,} | contested {int(df.contested.sum()):,}")
for name, g in [("all", df), ("handoff", df[df.in_handoff]), ("contested", df[df.contested])]:
    line = f"{name:10s} rows {len(g):>8,} pos {g.label.mean():.3f}"
    for m in ("qwen", "xl", "ce_full", "ce_large", "lgbm_full"):
        h = g[g[m].notna()]
        if len(h) and h.label.nunique() == 2:
            line += f" | {m} AUC {roc_auc_score(h.label, h[m]):.4f} ll {log_loss(h.label, np.clip(h[m], 1e-7, 1 - 1e-7), labels=[0, 1]):.4f} (n={len(h):,})"
    print(line)
