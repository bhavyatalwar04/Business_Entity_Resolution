"""Submission from the stacker (src.stack_eval): train one stacker on all fold-0 pairs of the handoff
sample, apply it to the test pairs (v2 LightGBM + CE test scores, competition computed over all test
S1s), decide, write, validate.

Usage: python -m src.stack_submit --out output/v5 --shift france:-1 [--features pool|meta|extra]
       [--extra ce_large=handoff/ce_large_out]
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
from src.stack_eval import STACK_PARAMS, build_X, load_ce

META = ["b_source", "len_name_a", "len_addr_a", "len_name_b", "len_addr_b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shift", default="")
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--features", default="pool", choices=["pool", "meta", "extra"])
    ap.add_argument("--extra", action="append", default=[], help="name=folder of an extra cross-encoder")
    args = ap.parse_args()
    extra = dict(e.split("=") for e in args.extra) if args.features == "extra" else {}
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    def attach(df, kind):
        df = df.merge(load_ce("handoff/ce_out", kind), on=["s1_id", "pool_id"], how="left")
        for name, d in extra.items():
            df = df.merge(load_ce(d, kind).rename(columns={"ce_prob": f"{name}_prob"}), on=["s1_id", "pool_id"],
                          how="left")
        return df.reset_index(drop=True)

    def select(X):
        return X if args.features != "pool" else X.drop(columns=META)

    # train on all fold-0 pairs (the only S1s with out-of-fold CE scores)
    pairs = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob("handoff/ce/train_pairs_part*.parquet"))],
                      ignore_index=True)
    tr = attach(pairs[(pairs["fold"] == 0) & (pairs["lgbm_prob"] >= 0.001)], "oof_fold0")
    for c in ["ce_prob"] + [f"{e}_prob" for e in extra]:
        tr[c] = tr[c].fillna(0.0)
    Xtr = select(build_X(tr, pairs[["s1_id", "pool_id", "lgbm_prob"]], "train", tuple(extra)))
    y = tr["label"].to_numpy()
    del pairs
    va = tr["s1_id"].map(lambda s: fold_of(s, 20) == 15).to_numpy()  # early-stopping slice
    m = lgb.train(STACK_PARAMS, lgb.Dataset(Xtr[~va], y[~va]), 3000, valid_sets=[lgb.Dataset(Xtr[va], y[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    m = lgb.train(STACK_PARAMS, lgb.Dataset(Xtr, y), m.best_iteration)
    print(f"stacker ({args.features}): {len(tr):,} training pairs, {Xtr.shape[1]} features, {m.num_trees()} trees",
          flush=True)

    # test
    lg = pd.read_parquet("artefacts/test/probs_model_v2.parquet").rename(columns={"prob": "lgbm_prob"})
    te = attach(lg[lg["lgbm_prob"] >= 0.001], "test")
    for c in ["ce_prob"] + [f"{e}_prob" for e in extra]:
        miss = int(te[c].isna().sum())
        print(f"test {c}: missing on {miss:,} of {len(te):,} pairs", flush=True)
        te[c] = te[c].fillna(te["lgbm_prob"])
    Xte = select(build_X(te, lg, "test", tuple(extra)))
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
        json.dump({"model": f"stack_{args.features}", "extra": extra, "decision": params_d, "shift": args.shift},
                  f, indent=1)


if __name__ == "__main__":
    main()
