"""inputs/france_all.parquet: every France test pair the Qwen CE scores = test candidates of France S1s with
LightGBM (full-data model) prob >= 0.001 (1,675,848 pairs). Same rule as the command used for the submission.
usage: python prep_france.py
"""
import glob

import pandas as pd

import paths as P

s1 = pd.read_csv(f"{P.DATA}/test/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3,
                 usecols=["entity_id", "country"])
fr = set(s1.entity_id[s1.country == "France"])
lg = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{P.INPUTS}/full/probs_model_full_part*"))])
f = lg[lg.s1_id.isin(fr) & (lg.prob >= 0.001)][["s1_id", "pool_id"]].reset_index(drop=True)
f.to_parquet(f"{P.INPUTS}/france_all.parquet", index=False)
print("france_all", len(f))
