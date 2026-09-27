"""handoff/ce_qwen_frcanon_out = v11's Qwen3 files with ONLY the France scores replaced by the canonicalised rescore.
ce_oof_fold0: copied from ce_qwen_out (India/US fold-0, unchanged); ce_test (contested): v11's scores, France rows
replaced; ce_test_france: the canonicalised France-all scores. So India/US must diff to 0 vs v11.
usage: python pack_frcanon.py <france_scores_dir> <worktree_handoff_dir>
"""
import glob, os, shutil, sys

import pandas as pd

FR, H = sys.argv[1], sys.argv[2]
NAME = sys.argv[3] if len(sys.argv) > 3 else "ce_qwen_frcanon_out"
src, dst = os.path.join(H, "ce_qwen_out"), os.path.join(H, NAME)
os.makedirs(dst, exist_ok=True)
rd = lambda p: pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(p))], ignore_index=True)
for f in glob.glob(os.path.join(src, "ce_oof_fold0_part*.parquet")) + [os.path.join(src, "ce_oof_fold0.DONE")]:
    shutil.copy(f, dst)
fr = rd(os.path.join(FR, "ce_test_france_part*.parquet"))[["s1_id", "pool_id", "ce_prob"]].astype({"ce_prob": "float32"})
te = rd(os.path.join(src, "ce_test_part*.parquet"))
m = te.merge(fr.rename(columns={"ce_prob": "fr"}), on=["s1_id", "pool_id"], how="left")
n_rep = int(m.fr.notna().sum())
m["ce_prob"] = m.fr.fillna(m.ce_prob).astype("float32")
m[["s1_id", "pool_id", "ce_prob"]].sort_values("s1_id", kind="stable").to_parquet(os.path.join(dst, "ce_test_part00.parquet"), compression="zstd", index=False)
open(os.path.join(dst, "ce_test.DONE"), "w").close()
fr.sort_values("s1_id", kind="stable").to_parquet(os.path.join(dst, "ce_test_france_part00.parquet"), compression="zstd", index=False)
open(os.path.join(dst, "ce_test_france.DONE"), "w").close()
print(f"{NAME}: fold-0 copied from v11 | contested test {len(m):,} rows, {n_rep:,} France rows replaced | France-all {len(fr):,}")
