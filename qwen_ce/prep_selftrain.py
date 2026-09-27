"""France self-training set for Qwen (laptop #258 FAIL branch), built to avoid r1's failure mode (loose teacher,
600k positives vs 56k negatives): labels only where ALL THREE scorers agree (lgbm_full, ce_full, Qwen),
and negatives kept at the labelled-train negative share, weighted toward hard ones (a model scored them > 0.02).

pos : lgbm, ce_full, qwen all >= 0.95 AND the pool record's best S1 with runner-up <= 0.5 (one-to-one)
neg : half one-to-one decoys (pool record owned by ANOTHER S1's 3-vote positive, this S1 >= 0.5 on a model), half all-three <= 0.05
mix : N_FR France pseudo pairs (neg share = TRAIN_NEG) + an equal number of labelled IN/US replay pairs (fold != 0)
out : inputs/selftrain_fr.parquet (text_a, text_b, label, kind) for qwen_dann.py --max_lambda 0 --src_file ...
"""
import glob, os, re

import numpy as np
import pandas as pd

I = os.path.expanduser("~/work/qwen_ce/inputs")
D = os.path.expanduser("~/work/dataset/dataset")
Q = os.path.expanduser("~/work/qwen_ce/out_scores")
N_FR, TRAIN_NEG, SEED = 150_000, 0.55, 5
rng = np.random.default_rng(SEED)
rd = lambda p: pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(p))], ignore_index=True)

lg = rd(f"{I}/full/probs_model_full_part*.parquet").rename(columns={"prob": "lgbm"})
ce = rd(f"{I}/ce_full/ce_test_part*.parquet").rename(columns={"ce_prob": "ce"})
qw = rd(f"{Q}/ce_test_france_part*.parquet").rename(columns={"ce_prob": "qwen"})
fr = qw.merge(lg, on=["s1_id", "pool_id"]).merge(ce, on=["s1_id", "pool_id"])
print(f"France pairs with all 3 scores: {len(fr):,}")

# one-to-one (hpc #260.1): combined score per pair; a pseudo-positive must be its pool record's BEST S1 with the
# runner-up S1 <= 0.5, so no pool record is a positive for two S1s
fr["comb"] = fr[["lgbm", "ce", "qwen"]].mean(axis=1)
rank = fr.sort_values("comb", ascending=False).groupby("pool_id").comb
fr["best_s1"] = fr.pool_id.map(fr.sort_values("comb", ascending=False).drop_duplicates("pool_id").set_index("pool_id").s1_id)
fr["second"] = fr.pool_id.map(rank.apply(lambda x: x.iloc[1] if len(x) > 1 else 0.0))
vote3 = (fr.lgbm >= 0.95) & (fr.ce >= 0.95) & (fr.qwen >= 0.95)
pos = fr[vote3 & (fr.s1_id == fr.best_s1) & (fr.second <= 0.5)]
# negatives (hpc #260.2): half one-to-one DECOYS = pool records that are a confident 3-vote positive of ANOTHER S1
# while THIS S1 scores them >= 0.5 on at least one model; half easy 3-vote negatives
owner = pos.set_index("pool_id").s1_id
decoy = fr[fr.pool_id.isin(owner.index) & (fr.s1_id != fr.pool_id.map(owner)) & (fr[["lgbm", "ce", "qwen"]].max(axis=1) >= 0.5)]
easy = fr[(fr.lgbm <= 0.05) & (fr.ce <= 0.05) & (fr.qwen <= 0.05)]
print(f"3-vote one-to-one positives {len(pos):,} | one-to-one decoys {len(decoy):,} | easy 3-vote negatives {len(easy):,}")
n_neg = int(N_FR * TRAIN_NEG); n_pos = N_FR - n_neg
nd = min(len(decoy), n_neg // 2)
negs = pd.concat([decoy.sample(nd, random_state=SEED), easy.sample(n_neg - nd, random_state=SEED)])
frs = pd.concat([pos.sample(min(len(pos), n_pos), random_state=SEED).assign(label=1),
                 negs.assign(label=0)])[["s1_id", "pool_id", "label"]]

lut = {}
for s in (1, 2, 3):
    t = pd.read_csv(f"{D}/test/test_source{s}.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
    lut.update(zip(t.entity_id, t.business_name + " | " + t.business_address))
fr_t = pd.DataFrame({"text_a": frs.s1_id.map(lut).values, "text_b": frs.pool_id.map(lut).values,
                     "label": frs.label.astype("int8").values, "kind": "fr_pseudo"})
tr = pd.read_parquet(f"{I}/train.parquet").sample(len(fr_t), random_state=SEED)
lut = {}
for s in (1, 2, 3):
    t = pd.read_csv(f"{D}/train/train_source{s}.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
    lut.update(zip(t.entity_id, t.business_name + " | " + t.business_address))
rep = pd.DataFrame({"text_a": tr.s1_id.map(lut).values, "text_b": tr.pool_id.map(lut).values,
                    "label": tr.label.astype("int8").values, "kind": "iu_replay"})
out = pd.concat([fr_t, rep], ignore_index=True)
assert out.text_a.notna().all() and out.text_b.notna().all()
out.to_parquet(f"{I}/selftrain_fr.parquet", index=False)
print(f"selftrain set {len(out):,}: France pseudo {len(fr_t):,} (pos {fr_t.label.mean():.3f}) + IN/US replay {len(rep):,} (pos {rep.label.mean():.3f})")
