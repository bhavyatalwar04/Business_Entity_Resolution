"""Add rachit #485 SIBLING hard negatives to inputs/selftrain_fr.parquet.
A pool record that is a confident one-to-one 3-vote positive of France S1-A, paired with a sibling S1-B (same core name =
name without legal-form tokens, different S1, different address) -> label 0. Replaces the same number of easy negatives.
"""
import glob, re

import numpy as np
import pandas as pd

SEED, N_SIB = 7, 42_000
D = "/home/rudra.120437/work/dataset/dataset"
rd = lambda p: pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(p))], ignore_index=True)
LEGAL = {"sarl", "sas", "sasu", "eurl", "sa", "snc", "sci", "cie", "5arl"}
core = lambda n: " ".join(t for t in re.sub(r"[^a-z0-9 ]", " ", n.lower()).split() if t not in LEGAL)
lut, s1name, s1addr = {}, {}, {}
for s in (1, 2, 3):
    t = pd.read_csv(f"{D}/test/test_source{s}.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
    lut.update(zip(t.entity_id, t.business_name + " | " + t.business_address))
    if s == 1:
        s1name.update(zip(t.entity_id, t.business_name)); s1addr.update(zip(t.entity_id, t.business_address))
fr = pd.read_parquet("inputs/france_all.parquet")[["s1_id", "pool_id"]]
lg = rd("inputs/full/probs_model_full_part*.parquet").rename(columns={"prob": "lgbm"})
ce = rd("inputs/ce_full/ce_test_part*.parquet").rename(columns={"ce_prob": "ce"})
qw = rd("out_scores/ce_test_france_part*.parquet").rename(columns={"ce_prob": "qwen"})
d = qw.merge(lg, on=["s1_id", "pool_id"]).merge(ce, on=["s1_id", "pool_id"])
pos = d[(d.lgbm >= 0.98) & (d.ce >= 0.98) & (d.qwen >= 0.98)]  # hpc #486.3: A assignment with all 3 teachers >= 0.98
pos = pos[pos.groupby("pool_id").s1_id.transform("size") == 1]            # the pool record is confidently one S1's
frs1 = pd.Series(sorted(set(fr.s1_id)))
cores = pd.DataFrame({"s1_id": frs1, "core": frs1.map(lambda x: core(s1name[x]))})
grp = cores.groupby("core").s1_id.apply(list)
sib = grp[grp.map(len) > 1]
p = pos.assign(core=pos.s1_id.map(dict(zip(cores.s1_id, cores.core))))
p = p[p.core.isin(sib.index)].sample(frac=1.0, random_state=SEED)
rng = np.random.default_rng(SEED); rows = []
for r in p.itertuples():
    others = [x for x in sib[r.core] if x != r.s1_id and s1addr[x] != s1addr[r.s1_id]]
    if others:
        rows.append((rng.choice(others), r.pool_id))
    if len(rows) >= N_SIB:
        break
neg = pd.DataFrame(rows, columns=["s1_id", "pool_id"])
# hpc #486.3: drop a sibling pair if ANY teacher scores (S1-B, pool) >= 0.5 (it could be B's real copy)
hi = d[(d[["lgbm", "ce", "qwen"]].max(axis=1) >= 0.5)].set_index(["s1_id", "pool_id"]).index
n0 = len(neg); neg = neg[~neg.set_index(["s1_id", "pool_id"]).index.isin(hi)]
print(f"sibling pairs dropped because a teacher scores them >= 0.5: {n0 - len(neg):,}")
neg = neg[~neg.set_index(["s1_id", "pool_id"]).index.isin(pos.set_index(["s1_id", "pool_id"]).index)]
st = pd.read_parquet("inputs/selftrain_fr.parquet")
fr_part, rep = st[st.kind == "fr_pseudo"], st[st.kind != "fr_pseudo"]
easy_idx = fr_part[fr_part.label == 0].index[: len(neg)]                   # replace easy negatives 1:1
sibdf = pd.DataFrame({"text_a": neg.s1_id.map(lut).values, "text_b": neg.pool_id.map(lut).values, "label": np.int8(0), "kind": "fr_sibling"})
out = pd.concat([fr_part.drop(easy_idx), sibdf, rep], ignore_index=True)
assert out.text_a.notna().all() and out.text_b.notna().all()
out.to_parquet("inputs/selftrain_fr.parquet", index=False)
print(f"sibling S1 groups {len(sib):,}; sibling negatives added {len(sibdf):,} (replacing easy negatives) | final {len(out):,}: {out.kind.value_counts().to_dict()} pos {out.label.mean():.3f}")
print(sibdf.head(3).to_string()[:900])
