"""France probes on top of v8: add France pairs that the best CE (ce_full) scores >= 0.95 but v8 left unmatched.
v8h: only pairs whose names carry conflicting legal forms (SARL vs SAS ...)   v8i: all such pairs.
Only pool records unmatched in v8 are used, one S1 per pool record (the highest-CE one), so one-to-one holds."""
import glob, os, numpy as np, pandas as pd
from src.features import _legal_forms
from src.io_utils import write_tsv
s1 = pd.read_parquet("artefacts/test/s1.parquet", columns=["entity_id", "name_clean", "country"])
pl = pd.concat([pd.read_parquet(f"artefacts/test/s{k}.parquet", columns=["entity_id", "name_clean"]) for k in (2, 3)])
n1, c1, np_ = dict(zip(s1["entity_id"], s1["name_clean"])), dict(zip(s1["entity_id"], s1["country"])), dict(zip(pl["entity_id"], pl["name_clean"]))
m = pd.read_csv("/home/arnav.119551/.claude/jobs/ea675fcc/tmp/diag/v8.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
pred = {a: [x for x in s.split(",") if x] for a, s in zip(m["source1_entity_id"], m["matched_entity_ids"])}
matched = set(x for v in pred.values() for x in v)
ce = pd.concat([pd.read_parquet(f) for f in glob.glob("handoff/ce_full_out/ce_test_part*.parquet")])
ce = ce[ce["s1_id"].map(c1).eq("France") & ~ce["pool_id"].isin(matched)]
hi = ce.sort_values("ce_prob", ascending=False).drop_duplicates("pool_id")
hi = hi[hi["ce_prob"] >= 0.95].copy()
def conflict(a, b):
    la, lb = _legal_forms(a), _legal_forms(b)
    return bool(la and lb and not (la & lb))
hi["legal_conflict"] = [conflict(n1[a], np_[b]) for a, b in zip(hi["s1_id"], hi["pool_id"])]
out_dir = "output/probes"; os.makedirs(out_dir, exist_ok=True)
s1_ids = m["source1_entity_id"].tolist()
for name, add in (("v8h_fr_legal", hi[hi["legal_conflict"]]), ("v8i_fr_cehigh", hi)):
    p = {k: list(v) for k, v in pred.items()}
    for a, b in zip(add["s1_id"], add["pool_id"]):
        p.setdefault(a, []).append(b)
    path = f"{out_dir}/{name}/matching_results.tsv"
    write_tsv(s1_ids, p, path, "matched_entity_ids")
    touched = add["s1_id"].nunique()
    print(f"{name}: +{len(add):,} pairs on {touched:,} France S1 ({touched / sum(c1[s] == 'France' for s in s1_ids) * 100:.1f}% of France S1)")
hi.to_parquet(f"{out_dir}/france_cehigh_rejected_pairs.parquet", index=False)
