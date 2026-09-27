"""Third Qwen training set (hpc #418): fold != 0 train pairs that neither train.parquet nor train_seed2.parquet contains.
700k contested (LightGBM prob in [0.005, 0.995]) + 175k easy (prob < 0.001 or > 0.999), shuffled -> inputs/train_seed3.parquet
"""
import glob
import pandas as pd

SEED = 21
rd = lambda p: pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(p))], ignore_index=True)
tp = rd("inputs/train_pairs_part*.parquet")
tp = tp[tp.fold != 0]
key = lambda d: d.s1_id.astype(str) + "|" + d.pool_id.astype(str)
seen = set(key(pd.read_parquet("inputs/train.parquet"))) | set(key(pd.read_parquet("inputs/train_seed2.parquet")))
tp = tp[~key(tp).isin(seen)]
hard = tp[(tp.lgbm_prob >= 0.005) & (tp.lgbm_prob <= 0.995)].sample(700_000, random_state=SEED)
easy = tp[(tp.lgbm_prob < 0.001) | (tp.lgbm_prob > 0.999)].sample(175_000, random_state=SEED)
out = pd.concat([hard, easy])[["s1_id", "pool_id", "label"]].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
out["label"] = out.label.astype("int8")
out.to_parquet("inputs/train_seed3.parquet", index=False)
print(f"train_seed3 {len(out):,} (pos {out.label.mean():.3f}): {len(hard):,} unseen contested (pos {hard.label.mean():.3f}) "
      f"+ {len(easy):,} unseen easy (pos {easy.label.mean():.3f}); overlap with train/seed2 = 0 by construction")
