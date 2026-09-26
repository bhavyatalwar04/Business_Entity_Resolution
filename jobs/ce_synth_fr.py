"""Fine-tune the France cross-encoder on synthetic labelled French pairs (src/synth_pairs.py), no external data.

Run this only if the India->US proxy (jobs/proxy_synth.py) shows that synthetic pairs help an unseen country.
init   : ce_france r1 (artefacts/ce_france/final), the scorer v8 already swaps in for unseen-country S1
train  : synthetic pairs from France test S1 texts (positives + generator-style decoys) mixed 1:1 with labelled
         handoff train pairs (fold != 0) so India/US behaviour is kept
output : drop-in replacement for handoff/ce_france_out:
         ce_oof_fold0_part*   labelled US/India fold 0 (no-harm check)
         ce_test_unseen_part* ALL France test pairs (handoff test pairs + step-3 candidates prob >= 0.001),
                              not only contested ones, so confidently wrong France pairs get a second opinion

Usage: PYTHONPATH=. python jobs/ce_synth_fr.py --out handoff/ce_synth_fr_out
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.ce_rescore import attach_text, load_texts, log, metrics, predict, read_pairs, train, write_parts
from src.io_utils import read_tsv
from src.synth_pairs import build_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="artefacts/ce_france/final")
    ap.add_argument("--ckpt", default="artefacts/ce_synth_fr")
    ap.add_argument("--out", default="handoff/ce_synth_fr_out")
    ap.add_argument("--anchors", type=int, default=150_000)
    ap.add_argument("--n_pos", type=int, default=2)
    ap.add_argument("--n_dec", type=int, default=2)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--infer_bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()
    args.model = args.init
    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    key = ["s1_id", "pool_id"]

    s1 = read_tsv("dataset/test/test_source1.tsv")
    seen = set(read_tsv("dataset/train/train_source1.tsv", usecols=["country"])["country"].unique())
    fr = s1[~s1["country"].isin(seen)]
    anchors = fr.sample(min(len(fr), args.anchors), random_state=args.seed)
    syn = build_pairs(anchors[["entity_id", "business_name", "business_address"]], "FR", rng, args.n_pos, args.n_dec)
    stats = {"unseen_s1": int(len(fr)), "anchors": int(len(anchors)), "synth_pairs": int(len(syn)),
             "synth_pos": int(syn["label"].sum()), "kinds": syn["kind"].value_counts().to_dict()}
    syn.to_parquet(os.path.join(args.out, "synth_fr.parquet"))
    log(f"synthetic France: {stats}")

    tr = read_pairs("handoff/ce/train_pairs_part*.parquet")
    tr = tr[tr["fold"] != 0]
    tr = tr[(tr["lgbm_prob"] >= 0.001) | (rng.rand(len(tr)) < 0.1)]
    tr = tr.iloc[rng.choice(len(tr), min(len(tr), len(syn)), replace=False)][key + ["label"]]
    mix = pd.concat([syn.assign(s1_id="te:" + syn["s1_id"])[key + ["label"]],
                     tr.assign(s1_id="tr:" + tr["s1_id"], pool_id="tr:" + tr["pool_id"])], ignore_index=True)
    stats["replay_pairs"] = int(len(tr))

    t_tr, t_te = load_texts("dataset", "train"), load_texts("dataset", "test")
    texts = pd.concat([pd.Series(t_tr.values, index="tr:" + t_tr.index), pd.Series(t_te.values, index="te:" + t_te.index),
                       pd.Series(syn["pool_text"].values, index=syn["pool_id"].values)])
    tok = AutoTokenizer.from_pretrained(args.init)
    model = train(args, tok, texts, tr=mix)
    del texts, mix

    f0 = read_pairs("handoff/ce/train_pairs_part*.parquet")
    f0 = f0[(f0["fold"] == 0) & (f0["lgbm_prob"] >= 0.001)].reset_index(drop=True)
    f0["ce_prob"] = predict(model, tok, *attach_text(f0, t_tr), args)
    m = {"n": int(len(f0)), "ce_synth_fr": metrics(f0["label"].values, f0["ce_prob"].values),
         "ce_large_reference": {"auc": 0.99839, "logloss": 0.04570}}
    json.dump(m, open(os.path.join(args.out, "fold0_metrics.json"), "w"), indent=1)
    log(f"fold-0 no-harm check {m}")
    write_parts(f0[key + ["ce_prob"]], args.out, "ce_oof_fold0")

    unseen = set(fr["entity_id"])
    h = read_pairs("handoff/ce/test_pairs_part*.parquet", columns=key)
    s3 = pd.read_parquet("artefacts/test/probs_model_full.parquet", columns=key + ["prob"])
    te = pd.concat([h[h["s1_id"].isin(unseen)], s3[s3["s1_id"].isin(unseen) & (s3["prob"] >= 0.001)][key]]).drop_duplicates()
    te = te.reset_index(drop=True)
    te["ce_prob"] = predict(model, tok, *attach_text(te, t_te), args)
    stats["france_pairs_scored"] = int(len(te))
    write_parts(te, args.out, "ce_test_unseen")
    json.dump(stats, open(os.path.join(args.out, "synth_stats.json"), "w"), indent=1)
    open(os.path.join(args.out, "ce_synth_fr.DONE"), "w").close()
    log(f"done {stats}")


if __name__ == "__main__":
    main()
