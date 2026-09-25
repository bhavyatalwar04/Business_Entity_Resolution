"""Submission from the stacker with pool-competition features (src.stack_eval 'stack_pool'):
train one stacker on all fold-0 pairs of the handoff sample, apply it to the test pairs
(v2 LightGBM + CE test scores, competition computed over all test S1s), decide, write, validate.

Usage: python -m src.stack_submit --out output/v4_stack_pool [--shift france:-1] [--alpha 1.5]
"""
import argparse
import glob
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from src.blend_eval import logit
from src.decide import decide
from src.evaluate import fold_of
from src.run import art_dir, write_submission
from src.stack_eval import pool_context, stack_features


def features(df, all_pairs, split):
    """Stacker matrix: S1-side context + empty-address flag + pool competition."""
    X = stack_features(df)
    pool = pd.concat([pd.read_parquet(f"artefacts/{split}/s{k}.parquet", columns=["entity_id", "addr_clean"])
                      for k in (2, 3)])
    empty = set(pool.loc[pool["addr_clean"].str.len() == 0, "entity_id"])
    X["b_empty_addr"] = df["pool_id"].isin(empty).to_numpy().astype(np.float32)
    return X.join(pool_context(all_pairs, df)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shift", default="")
    ap.add_argument("--alpha", type=float, default=1.5)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # train on all fold-0 pairs (the only S1s with out-of-fold CE scores)
    pairs = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob("handoff/ce/train_pairs_part*.parquet"))],
                      ignore_index=True)
    ce = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob("handoff/ce_out/ce_oof_fold0_part*.parquet"))],
                   ignore_index=True)
    tr = pairs[(pairs["fold"] == 0) & (pairs["lgbm_prob"] >= 0.001)].merge(ce, on=["s1_id", "pool_id"], how="left")
    tr = tr.fillna({"ce_prob": 0.0}).reset_index(drop=True)
    Xtr = features(tr, pairs[["s1_id", "pool_id", "lgbm_prob"]], "train")
    y = tr["label"].to_numpy()
    del pairs
    va = tr["s1_id"].map(lambda s: fold_of(s, 20) == 15).to_numpy()  # early-stopping slice
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100,
              "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 42}
    m = lgb.train(params, lgb.Dataset(Xtr[~va], y[~va]), 3000, valid_sets=[lgb.Dataset(Xtr[va], y[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    m = lgb.train(params, lgb.Dataset(Xtr, y), m.best_iteration)
    print(f"stacker: {len(tr):,} training pairs, {m.num_trees()} trees", flush=True)

    # test
    lg = pd.read_parquet("artefacts/test/probs_model_v2.parquet").rename(columns={"prob": "lgbm_prob"})
    ce_t = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob("handoff/ce_out/ce_test_part*.parquet"))],
                     ignore_index=True)
    te = lg[lg["lgbm_prob"] >= 0.001].merge(ce_t, on=["s1_id", "pool_id"], how="left").reset_index(drop=True)
    te["ce_prob"] = te["ce_prob"].fillna(te["lgbm_prob"])
    Xte = features(te, lg, "test")
    del lg
    te["prob"] = m.predict(Xte[Xtr.columns])
    s1 = pd.read_parquet(os.path.join(art_dir(cfg, "test"), "s1.parquet"), columns=["entity_id", "country_norm"])
    if args.shift:
        c, delta = args.shift.split(":")
        in_c = te["s1_id"].isin(set(s1.loc[s1["country_norm"] == c, "entity_id"])).to_numpy()
        z = logit(te["prob"].to_numpy())
        te["prob"] = np.where(in_c, 1 / (1 + np.exp(-(z + float(delta)))), te["prob"].to_numpy())
        print(f"shifted logit by {delta} on {in_c.sum():,} pairs of country {c}", flush=True)
    params_d = {"method": "expf", "alpha": args.alpha, "one_to_one": True}
    s1_ids = s1["entity_id"].tolist()
    write_submission(cfg, s1_ids, decide(te[["s1_id", "pool_id", "prob"]], s1_ids, params_d), args.out)
    with open(os.path.join(args.out, "blend.json"), "w") as f:
        json.dump({"model": "stack_pool", "decision": params_d, "shift": args.shift}, f, indent=1)


if __name__ == "__main__":
    main()
