"""LightGBM pair scorer.

Training uses folds grouped by S1 entity (fold = crc32(s1_id) % n_folds), so every
candidate of an S1 is either in train or in validation - the OOF probabilities are honest
and are what the decision layer is tuned on. The fold models are all kept and averaged at
inference time.
"""
import os

import lightgbm as lgb
import numpy as np

from src.evaluate import fold_of

ID_COLS = ["s1_id", "pool_id"]


def lgb_params(cfg):
    """LightGBM parameters from the config."""
    c = cfg["lgbm"]
    return {
        "objective": "binary", "learning_rate": c["learning_rate"], "num_leaves": c["num_leaves"],
        "min_data_in_leaf": c.get("min_data_in_leaf", 50), "feature_fraction": c.get("feature_fraction", 0.8),
        "bagging_fraction": c.get("bagging_fraction", 0.8), "bagging_freq": 1, "lambda_l2": 1.0,
        "verbose": -1, "seed": cfg["seed"], "num_threads": -1,
    }


def train_lgbm(X, y, s1_ids, feat_cols, cfg):
    """Train with folds grouped by S1 id on a float32 matrix X; return (fold models, OOF, feat_cols).
    Each fold's LightGBM Dataset is constructed immediately so its row slice can be freed
    (keeps peak memory near one copy of X - needed for 10M+ rows on 16 GB)."""
    y = np.asarray(y, dtype=np.int8)
    n_folds = cfg["validation"]["n_folds"]
    fold_map = {s: fold_of(s, n_folds) for s in pd_unique(s1_ids)}
    fold = np.array([fold_map[s] for s in s1_ids], dtype=np.int8)
    oof = np.zeros(len(y))
    models = []
    params = lgb_params(cfg)
    for k in range(n_folds):
        tr, va = fold != k, fold == k
        Xtr = X[tr]
        dtr = lgb.Dataset(Xtr, y[tr], feature_name=feat_cols, free_raw_data=True).construct()
        del Xtr
        Xva = X[va]
        dva = lgb.Dataset(Xva, y[va], reference=dtr).construct()
        m = lgb.train(params, dtr, num_boost_round=cfg["lgbm"]["n_estimators"], valid_sets=[dva],
                      callbacks=[lgb.early_stopping(cfg["lgbm"]["early_stopping_rounds"], verbose=False)])
        oof[va] = m.predict(Xva, num_iteration=m.best_iteration)
        del dtr, dva, Xva
        models.append(m)
        print(f"  fold {k}: best_iter {m.best_iteration}, val logloss {m.best_score['valid_0']['binary_logloss']:.4f}",
              flush=True)
    imp = np.mean([m.feature_importance("gain") for m in models], axis=0)
    top = sorted(zip(feat_cols, imp), key=lambda x: -x[1])[:15]
    print("  top features: " + ", ".join(n for n, _ in top), flush=True)
    return models, oof, feat_cols


def pd_unique(a):
    """Order-preserving unique values of an array (pandas' hash-based unique)."""
    import pandas as pd
    return pd.unique(a)


def predict_lgbm(features, model_dir, model_cfg, stage=1):
    """Average probability of the saved fold models of the given stage."""
    cols = model_cfg["features"] if stage == 1 else model_cfg["features2"]
    prefix = "lgbm" if stage == 1 else "lgbm2"
    X = features[cols].to_numpy(np.float32)
    probs = np.zeros(len(X))
    for i in range(model_cfg["n_models"]):
        m = lgb.Booster(model_file=os.path.join(model_dir, f"{prefix}_{i}.txt"))
        probs += m.predict(X)
    return probs / model_cfg["n_models"]


def prob_context(df, prob):
    """Stage-2 features from stage-1 probabilities, S1 side only (identical on train sample and
    test): rank, gap to best, second best, probability mass and count above 0.5 per S1."""
    import pandas as pd
    d = pd.DataFrame({"s1": df["s1_id"].to_numpy(), "p": np.asarray(prob, dtype=np.float32)})
    d["hi"] = (d["p"] > 0.5).astype(np.float32)
    g = d.groupby("s1", sort=False)
    rank = g["p"].rank(ascending=False, method="first")
    out = pd.DataFrame({"p1": d["p"].to_numpy()})
    out["p1_rank"] = rank.to_numpy(np.float32)
    out["p1_max"] = g["p"].transform("max").to_numpy(np.float32)
    out["p1_gap"] = out["p1_max"] - out["p1"]
    out["p1_sum"] = g["p"].transform("sum").to_numpy(np.float32)
    out["p1_n50"] = g["hi"].transform("sum").to_numpy(np.float32)
    second = d.loc[rank.to_numpy() == 2].set_index("s1")["p"]
    out["p1_second"] = d["s1"].map(second).fillna(0.0).to_numpy(np.float32)
    return out


def stage2_matrix(X, ids_df, prob1):
    """Stage-2 design matrix = stage-1 features + prob_context columns; returns (X2, context column names)."""
    ctx = prob_context(ids_df, prob1)
    return np.hstack([X, ctx.to_numpy(np.float32)]), list(ctx.columns)
