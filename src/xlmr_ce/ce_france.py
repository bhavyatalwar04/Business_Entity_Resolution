"""Transductive self-training of the cross-encoder on unseen-country test records (pseudo-labels only, no external data).

Countries are an open set: "unseen" = test S1 whose country label never occurs in train_source1.
Pseudo-labels come from our own models on the provided test candidates (handoff test pairs):
  blend = sigmoid(0.2*logit(lgbm_v2) + 0.4*logit(ce_base) + 0.4*logit(ce_large)), one-to-one per pool record.
  positive : blend >= POS, every model >= 0.8, pair is the pool record's best S1, runner-up S1 blend <= 0.5
  negative : every model <= 0.2; decoys first (pool record is a confident positive of ANOTHER S1), then random fill
Training mixes the pseudo-labelled pairs with an equal number of labelled train pairs (handoff fold != 0), and continues
from the fine-tuned xlm-roberta-large checkpoint at a low learning rate.
Outputs (<out>): ce_oof_fold0_part* (sanity check on labelled US/India fold 0), ce_test_unseen_part* (all test pairs of
unseen-country S1), pseudo_stats.json, fold0_metrics.json.

Usage: PYTHONPATH=. python -m src.xlmr_ce.ce_france --init artefacts/ce_large/final --ckpt artefacts/ce_france --out handoff/ce_france_out
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.xlmr_ce.ce_rescore import attach_text, load_texts, log, metrics, predict, read_pairs, train, write_parts
from src.io_utils import read_tsv

EPS = 1e-6


def logit(p):
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="artefacts/ce_large/final")
    ap.add_argument("--ckpt", default="artefacts/ce_france")
    ap.add_argument("--out", default="handoff/ce_france_out")
    ap.add_argument("--pos", type=float, default=0.97)
    ap.add_argument("--neg_per_pos", type=float, default=1.5)
    ap.add_argument("--cap_pos", type=int, default=600_000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--infer_bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--save_every", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--round2", action="store_true")
    args = ap.parse_args()
    args.model = args.init
    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    stats = {}

    # --- unseen-country test S1 (open set)
    s1 = read_tsv("dataset/test/test_source1.tsv", usecols=["entity_id", "country"])
    seen = set(read_tsv("dataset/train/train_source1.tsv", usecols=["country"])["country"].unique())
    unseen = set(s1.loc[~s1["country"].isin(seen), "entity_id"])
    stats["unseen_countries"] = sorted(set(s1["country"]) - seen)
    stats["unseen_s1"] = len(unseen)

    # --- test pairs of unseen S1 with all three model scores
    key = ["s1_id", "pool_id"]
    if args.round2:
        # union of handoff test pairs and step-3 candidates (prob >= 0.001); signals: step-3 LightGBM, ce_large, ce_france r1
        h = read_pairs("handoff/ce/test_pairs_part*.parquet", columns=key)
        s3 = pd.read_parquet("artefacts/test/probs_model_full.parquet", columns=key + ["prob"])
        s3 = s3[s3["s1_id"].isin(unseen)]
        te = pd.concat([h[h["s1_id"].isin(unseen)], s3[s3["prob"] >= 0.001][key]]).drop_duplicates()
        te = te.merge(s3.rename(columns={"prob": "lgbm_prob"}), on=key, how="left").fillna({"lgbm_prob": 0.0005})
        big = pd.concat([read_pairs("handoff/ce_large_out/ce_test_part*.parquet"),
                         read_pairs("handoff/ce_large_out/ce_extra_test_part*.parquet")]).drop_duplicates(key)
        fr = pd.concat([read_pairs("handoff/ce_france_out/ce_test_unseen_part*.parquet"),
                        read_pairs("handoff/ce_france_out/ce_extra_test_part*.parquet")]).drop_duplicates(key)
        te = te.merge(big.rename(columns={"ce_prob": "base"}), on=key)      # "base" slot = ce_large here
        te = te.merge(fr.rename(columns={"ce_prob": "large"}), on=key)      # "large" slot = ce_france r1 here
        w = (0.3, 0.35, 0.35)
    else:
        te = read_pairs("handoff/ce/test_pairs_part*.parquet")
        te = te[te["s1_id"].isin(unseen)]
        te = te.merge(read_pairs("handoff/ce_out/ce_test_part*.parquet").rename(columns={"ce_prob": "base"}), on=key)
        te = te.merge(read_pairs("handoff/ce_large_out/ce_test_part*.parquet").rename(columns={"ce_prob": "large"}), on=key)
        w = (0.2, 0.4, 0.4)
    te["blend"] = 1 / (1 + np.exp(-(w[0] * logit(te["lgbm_prob"].values) + w[1] * logit(te["base"].values)
                                    + w[2] * logit(te["large"].values))))
    stats["unseen_pairs"] = len(te)
    stats["signals"] = "lgbm_full, ce_large, ce_france_r1" if args.round2 else "lgbm_v2, ce_base, ce_large"
    log(f"unseen-country S1 {len(unseen):,} ({stats['unseen_countries']}), pairs {len(te):,}")

    # pool record's best and runner-up S1
    te = te.sort_values(["pool_id", "blend"], ascending=[True, False]).reset_index(drop=True)
    first = ~te["pool_id"].duplicated()
    runner = te["blend"].where(~first).groupby(te["pool_id"]).transform("max").fillna(0.0)
    mn = te[["lgbm_prob", "base", "large"]].min(axis=1)
    mx = te[["lgbm_prob", "base", "large"]].max(axis=1)
    pos = first & (te["blend"] >= args.pos) & (mn >= 0.8) & (runner <= 0.5)
    pos_idx = np.flatnonzero(pos.values)
    if len(pos_idx) > args.cap_pos:
        pos_idx = rng.choice(pos_idx, args.cap_pos, replace=False)
    confident_pool = set(te.loc[pos, "pool_id"])
    negok = (mx <= 0.2).values
    decoy = negok & ~first.values & te["pool_id"].isin(confident_pool).values
    n_neg = int(args.neg_per_pos * len(pos_idx))
    d_idx = np.flatnonzero(decoy)
    if len(d_idx) > n_neg // 2:
        d_idx = rng.choice(d_idx, n_neg // 2, replace=False)
    r_pool = np.flatnonzero(negok & ~decoy)
    r_idx = rng.choice(r_pool, min(len(r_pool), n_neg - len(d_idx)), replace=False)
    pl = pd.concat([te.iloc[pos_idx][key].assign(label=1), te.iloc[np.r_[d_idx, r_idx]][key].assign(label=0)])
    stats.update(pseudo_pos=int(len(pos_idx)), pseudo_decoy_neg=int(len(d_idx)), pseudo_rand_neg=int(len(r_idx)),
                 pos_candidates=int(pos.sum()), decoy_candidates=int(decoy.sum()))
    log(f"pseudo-labels: {stats}")
    pl = pl.assign(s1_id="te:" + pl["s1_id"], pool_id="te:" + pl["pool_id"])

    # --- labelled replay (handoff fold != 0, same selection as the original CE training)
    tr = read_pairs("handoff/ce/train_pairs_part*.parquet")
    tr = tr[tr["fold"] != 0]
    tr = tr[(tr["lgbm_prob"] >= 0.001) | (rng.rand(len(tr)) < 0.1)]
    tr = tr.iloc[rng.choice(len(tr), min(len(tr), len(pl)), replace=False)][key + ["label"]]
    tr = tr.assign(s1_id="tr:" + tr["s1_id"], pool_id="tr:" + tr["pool_id"])
    mix = pd.concat([pl, tr], ignore_index=True)
    stats.update(replay_pairs=int(len(tr)), train_pairs=int(len(mix)))

    t_tr, t_te = load_texts("dataset", "train"), load_texts("dataset", "test")
    texts = pd.concat([pd.Series(t_tr.values, index="tr:" + t_tr.index), pd.Series(t_te.values, index="te:" + t_te.index)])
    tok = AutoTokenizer.from_pretrained(args.init)
    model = train(args, tok, texts, tr=mix)
    del texts, mix

    # --- sanity check on labelled fold 0 (US/India)
    f0 = read_pairs("handoff/ce/train_pairs_part*.parquet")
    f0 = f0[(f0["fold"] == 0) & (f0["lgbm_prob"] >= 0.001)].reset_index(drop=True)
    f0["ce_prob"] = predict(model, tok, *attach_text(f0, t_tr), args)
    m = {"n": int(len(f0)), "ce_france": metrics(f0["label"].values, f0["ce_prob"].values),
         "ce_large_reference": {"auc": 0.99839, "logloss": 0.04570}}
    json.dump(m, open(os.path.join(args.out, "fold0_metrics.json"), "w"), indent=1)
    log(f"fold-0 sanity {m}")
    write_parts(f0[key + ["ce_prob"]], args.out, "ce_oof_fold0")

    # --- all test pairs of unseen-country S1
    out = te[key].copy()
    out["ce_prob"] = predict(model, tok, *attach_text(out, t_te), args)
    stats["mean_prob_change_vs_prev"] = float((out["ce_prob"].values - te["large"].values).mean())
    write_parts(out, args.out, "ce_test_unseen")
    json.dump(stats, open(os.path.join(args.out, "pseudo_stats.json"), "w"), indent=1)
    open(os.path.join(args.out, "ce_france.DONE"), "w").close()
    log(f"done {stats}")


if __name__ == "__main__":
    main()
