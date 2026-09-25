"""Stacker over LightGBM + CE scores with per-S1 context, evaluated on the fold-0 S1s of the handoff
sample with the same cross-fitted protocol as src.blend_eval (so scores are directly comparable).

Stacker OOF: fold-0 S1s are split in 4 quarters (crc32 % 20 in {0, 5, 10, 15}); each quarter is
predicted by a small LightGBM trained on the other three. The decision is then tuned / scored on the
two halves exactly as in blend_eval.

Usage: python -m src.stack_eval
"""
import glob
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.blend_eval import held_out, logit
from src.io_utils import load_ground_truth


def context(df, col, prefix):
    """Per-S1 rank, gap to best, second best, sum and count > 0.5 of score column col."""
    g = df.groupby("s1_id", sort=False)[col]
    out = pd.DataFrame(index=df.index)
    rank = g.rank(ascending=False, method="first")
    out[f"{prefix}_rank"] = rank
    mx = g.transform("max")
    out[f"{prefix}_gap"] = mx - df[col]
    out[f"{prefix}_max"] = mx
    out[f"{prefix}_sum"] = g.transform("sum")
    out[f"{prefix}_n50"] = (df[col] > 0.5).groupby(df["s1_id"], sort=False).transform("sum")
    second = df.loc[rank == 2].set_index("s1_id")[col]
    out[f"{prefix}_second"] = df["s1_id"].map(second).fillna(0.0)
    return out


def pool_context(all_pairs, df):
    """Candidate-side competition from LightGBM probabilities over ALL sampled S1s (every fold):
    how many S1s compete for the pool record, this S1's rank among them, and the best other S1's prob."""
    a = all_pairs[["s1_id", "pool_id", "lgbm_prob"]]
    a = a[a["lgbm_prob"] >= 0.001]
    g = a.groupby("pool_id")["lgbm_prob"]
    top2 = g.nlargest(2).groupby(level=0).agg(list)
    first = top2.map(lambda v: v[0])
    second = top2.map(lambda v: v[1] if len(v) > 1 else 0.0)
    n = g.size()
    out = pd.DataFrame(index=df.index)
    out["pool_n"] = df["pool_id"].map(n).fillna(1).to_numpy()
    f1, f2 = df["pool_id"].map(first).to_numpy(), df["pool_id"].map(second).to_numpy()
    p = df["lgbm_prob"].to_numpy()
    out["pool_best_other"] = np.where(p >= f1, f2, f1)
    out["pool_margin"] = p - out["pool_best_other"]
    out["pool_is_top"] = (p >= f1).astype(np.float32)
    return out


def stack_features(df):
    """Pair-level stacker inputs."""
    X = pd.DataFrame({"lgbm": df["lgbm_prob"], "ce": df["ce_prob"],
                      "lgbm_logit": logit(df["lgbm_prob"].to_numpy()), "ce_logit": logit(df["ce_prob"].to_numpy())},
                     index=df.index)
    X["blend"] = 1 / (1 + np.exp(-(0.5 * X["lgbm_logit"] + 0.5 * X["ce_logit"])))
    X["n_cands"] = df.groupby("s1_id", sort=False)["pool_id"].transform("size")
    for col, p in (("lgbm_prob", "l"), ("ce_prob", "c")):
        X = X.join(context(df, col, p))
    tmp = df[["s1_id"]].assign(blend=X["blend"].to_numpy())
    X = X.join(context(tmp, "blend", "b"))
    return X.astype(np.float32)


def main():
    gold = load_ground_truth("dataset")
    pairs = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob("handoff/ce/train_pairs_part*.parquet"))],
                      ignore_index=True)
    pairs = pairs[pairs["fold"] == 0].drop(columns=["fold"])
    ids = list(pd.unique(pairs["s1_id"]))
    ce = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob("handoff/ce_out/ce_oof_fold0_part*.parquet"))],
                   ignore_index=True)
    df = pairs[pairs["lgbm_prob"] >= 0.001].merge(ce, on=["s1_id", "pool_id"], how="left").reset_index(drop=True)
    df["ce_prob"] = df["ce_prob"].fillna(0.0)
    X = stack_features(df)
    X_s1 = X.copy()
    pool = pd.concat([pd.read_parquet(f"artefacts/train/s{k}.parquet", columns=["entity_id", "addr_clean"])
                      for k in (2, 3)])
    empty = set(pool.loc[pool["addr_clean"].str.len() == 0, "entity_id"])
    X["b_empty_addr"] = df["pool_id"].isin(empty).to_numpy().astype(np.float32)
    X = X.join(pool_context(pd.concat([pd.read_parquet(p, columns=["s1_id", "pool_id", "lgbm_prob"]) for p in
                                       sorted(glob.glob("handoff/ce/train_pairs_part*.parquet"))]), df)).astype(np.float32)
    y = df["label"].to_numpy()
    h = {s: zlib.crc32(s.encode()) % 20 for s in ids}
    q = df["s1_id"].map(h).to_numpy()
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100,
              "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 42}
    def oof_stack(M):
        oof = np.zeros(len(df))
        for k in (0, 5, 10, 15):
            te = q == k
            # early stopping on one of the three training quarters
            va_q = [x for x in (0, 5, 10, 15) if x != k][0]
            fit, va = ~te & (q != va_q), q == va_q
            m = lgb.train(params, lgb.Dataset(M[fit], y[fit]), 3000, valid_sets=[lgb.Dataset(M[va], y[va])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[te] = m.predict(M[te], num_iteration=m.best_iteration)
        return oof

    df["stack"] = oof_stack(X_s1)
    df["stack_pool"] = oof_stack(X)
    df["logit_avg_w0.5"] = X["blend"].to_numpy()
    s1 = pd.read_parquet("artefacts/train/s1.parquet", columns=["entity_id", "country_norm"])
    country = dict(zip(s1["entity_id"], s1["country_norm"]))
    hh = {s: zlib.crc32(s.encode()) % 10 for s in ids}
    halves = [[s for s in ids if hh[s] == 0], [s for s in ids if hh[s] == 5]]
    for name in ("logit_avg_w0.5", "stack", "stack_pool"):
        score, by_c, p = held_out(df, name, gold, halves, country)
        print(f"  {name:16s} held-out macro F0.5 {score:.4f} | " +
              " | ".join(f"{c} {x:.4f}" for c, x in sorted(by_c.items())) + f" | params(all) {p}", flush=True)


if __name__ == "__main__":
    main()
