"""Pack one new cross-encoder's outputs into the CE handoff format and compare it with Qwen3 (v11) on the same fold-0 rows.

usage: python pack_model.py <out_dir> <france_scores_dir> <handoff_dir> "<model description>"
  <out_dir>/qwen_oof_fold0.parquet, qwen_test_contested.parquet  (written by qwen_ce.py)
  <france_scores_dir>/ce_test_france_part*.parquet + ce_test_france.DONE  (written by qwen_score.py; optional)
writes <handoff_dir>/ce_oof_fold0_part*, ce_test_part*, ce_test_france_part* (s1_id, pool_id, ce_prob float32, sorted by
s1_id, zstd, < 90 MB each) + .DONE markers + fold0_metrics.json (all / handoff / contested AUC, new model vs Qwen3).
Contested = sigmoid(0.5 logit(lgbm_full) + 0.5 logit(ce_full)) in [0.02, 0.98], as in contested_auc.py.
"""
import glob, json, os, sys

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

OUT, FR, DST, DESC = sys.argv[1:5]
I = os.path.expanduser("~/work/qwen_ce/inputs")
Q3 = os.path.expanduser("~/work/qwen_ce/out/qwen_oof_fold0.parquet")
ROWS_PER_PART = 3_000_000
os.makedirs(DST, exist_ok=True)
rd = lambda p: pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(p))], ignore_index=True)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
ll = lambda y, p: float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))


def parts(df, stem):
    df = df[["s1_id", "pool_id", "ce_prob"]].astype({"ce_prob": "float32"}).sort_values("s1_id", kind="stable")
    assert df.ce_prob.notna().all(), f"{stem} has NaN scores"
    for i, s in enumerate(range(0, len(df), ROWS_PER_PART)):
        f = os.path.join(DST, f"{stem}_part{i:02d}.parquet")
        df.iloc[s:s + ROWS_PER_PART].to_parquet(f, compression="zstd", index=False)
        mb = os.path.getsize(f) / 2**20
        assert mb < 90, f"{f} is {mb:.1f} MB"
        print(f"{f}  {min(ROWS_PER_PART, len(df) - s):,} rows  {mb:.1f} MB")
    open(os.path.join(DST, f"{stem}.DONE"), "w").close()


oof = pd.read_parquet(os.path.join(OUT, "qwen_oof_fold0.parquet"))
parts(oof, "ce_oof_fold0")
df = pd.read_parquet(f"{I}/fold0.parquet")[["s1_id", "pool_id", "label", "in_handoff"]].merge(oof, on=["s1_id", "pool_id"])
df = df.merge(pd.read_parquet(Q3).rename(columns={"ce_prob": "qwen3"}), on=["s1_id", "pool_id"], how="left")
df = df.merge(rd(f"{I}/ce_full/ce_oof_fold0_part*.parquet").rename(columns={"ce_prob": "ce_full"}), on=["s1_id", "pool_id"], how="left")
o = rd(f"{I}/full/oof_train_full_part*.parquet")
df = df.merge(o[o.fold == 0][["s1_id", "pool_id", "prob"]].rename(columns={"prob": "lgbm_full"}), on=["s1_id", "pool_id"], how="left")
z = 1 / (1 + np.exp(-(0.5 * lg(df.lgbm_full.fillna(0.0)) + 0.5 * lg(df.ce_full.fillna(0.0)))))
df["contested"] = (z >= 0.02) & (z <= 0.98) & df.lgbm_full.notna() & df.ce_full.notna()
metrics = {"model": DESC}
for name, g in [("all", df), ("handoff", df[df.in_handoff]), ("contested", df[df.contested])]:
    g = g[g.qwen3.notna()]
    avg = 1 / (1 + np.exp(-(0.5 * lg(g.ce_prob.values) + 0.5 * lg(g.qwen3.values))))  # 2-adapter logit average with v11's Qwen3
    metrics[name] = {"rows": int(len(g)), "auc": float(roc_auc_score(g.label, g.ce_prob)), "logloss": ll(g.label, g.ce_prob),
                     "qwen3_auc": float(roc_auc_score(g.label, g.qwen3)), "qwen3_logloss": ll(g.label, g.qwen3),
                     "avg_with_qwen3_auc": float(roc_auc_score(g.label, avg)),
                     "logit_corr_with_qwen3": float(np.corrcoef(lg(g.ce_prob.values), lg(g.qwen3.values))[0, 1])}
json.dump(metrics, open(os.path.join(DST, "fold0_metrics.json"), "w"), indent=1)
print(json.dumps(metrics, indent=1))

if os.path.exists(os.path.join(OUT, "qwen_test_contested.parquet")):
    parts(pd.read_parquet(os.path.join(OUT, "qwen_test_contested.parquet")), "ce_test")
else:
    print("contested test scores not there yet:", OUT)
if os.path.exists(os.path.join(FR, "ce_test_france.DONE")):
    parts(rd(os.path.join(FR, "ce_test_france_part*.parquet")), "ce_test_france")
else:
    print("France-all scores not there yet:", FR)
