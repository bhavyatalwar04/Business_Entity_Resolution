"""CLI entry point: python -m src.run --stage all --split test

Every stage caches its output under artefacts/<split>/ so a single stage
(e.g. --stage decide) can be rerun without redoing the ones before it.
"""
import argparse
import os
import time

import pandas as pd
import yaml

from src.io_utils import read_tsv
from src.normalize import normalize_frame

STAGES = ["normalize", "embed", "block", "features", "cross_encoder", "rank", "decide"]

KEEP_COLS = ["entity_id", "business_name", "business_address", "country", "source",
             "name_clean", "name_core", "name_skel", "name_ns", "name_alt", "name_alt_skel",
             "addr_clean", "postal", "nums", "landmarks", "country_norm"]


def parse_args():
    """Parse --stage, --split and --config command-line options."""
    parser = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    parser.add_argument("--stage", default="all", choices=["all", *STAGES])
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def art_dir(cfg, split):
    """artefacts/<split>/, created on demand."""
    d = os.path.join(cfg["paths"]["artefacts_dir"], split)
    os.makedirs(d, exist_ok=True)
    return d


def stage_normalize(cfg, split, chunk_rows=1_000_000):
    """Clean each source TSV in row chunks (bounded memory) into artefacts/<split>/s{k}.parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    t0 = time.time()
    d = art_dir(cfg, split)
    for k in (1, 2, 3):
        src = os.path.join(cfg["paths"]["data_dir"], split, f"{split}_source{k}.tsv")
        raw = read_tsv(src)
        raw["source"] = k
        writer = None
        for start in range(0, len(raw), chunk_rows):
            out = normalize_frame(raw.iloc[start:start + chunk_rows])[KEEP_COLS]
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(os.path.join(d, f"s{k}.parquet"), table.schema)
            writer.write_table(table)
        writer.close()
        print(f"[normalize:{split}] source{k}: {len(raw):,} rows [{time.time() - t0:.0f}s]", flush=True)
        del raw


def load_normalized(cfg, split, columns=None):
    """Load cached normalised frames: (s1, pool) where pool = Source 2 + Source 3."""
    d = art_dir(cfg, split)
    s1 = pd.read_parquet(os.path.join(d, "s1.parquet"), columns=columns)
    pool = pd.concat([pd.read_parquet(os.path.join(d, f"s{k}.parquet"), columns=columns) for k in (2, 3)],
                     ignore_index=True)
    return s1, pool


def stage_embed(cfg, split):
    """Encode S1 and pool records with the blocking encoder; cache emb_s1.npy / emb_pool.npy."""
    import numpy as np
    from src.embed import encode, record_text
    d = art_dir(cfg, split)
    model = cfg["blocking"].get("embed_model_path") or cfg["blocking"]["embed_model"]
    s1, pool = load_normalized(cfg, split, columns=["name_clean", "addr_clean"])
    for name, df in (("s1", s1), ("pool", pool)):
        path = os.path.join(d, f"emb_{name}.npy")
        if os.path.exists(path):
            print(f"[embed:{split}] {name}: cached, skipping", flush=True)
            continue
        t0 = time.time()
        tmp = path.replace(".npy", ".tmp.npy")
        emb = encode(record_text(df), model_name=model, out_path=tmp)
        shape = emb.shape
        del emb
        os.replace(tmp, path)
        print(f"[embed:{split}] {name} {shape} [{time.time() - t0:.0f}s]", flush=True)


NORM_COLS = ["entity_id", "source", "name_clean", "name_core", "name_skel", "name_ns", "name_alt",
             "name_alt_skel", "addr_clean", "postal", "nums", "landmarks", "country_norm"]


def query_index(cfg, split, s1):
    """S1 rows the matcher is trained / scored on: all of test, a fixed random sample of train.
    (Blocking always runs on ALL S1 so candidate-competition context matches test.)"""
    import numpy as np
    if split == "test":
        return np.arange(len(s1))
    n = min(cfg["train"]["sample_s1"], len(s1))
    return np.sort(np.random.RandomState(cfg["seed"]).choice(len(s1), n, replace=False))


def stage_block(cfg, split):
    """Token + embedding blocking for the query S1s against the full pool -> cands.parquet."""
    import numpy as np
    from src.blocking import token_block, union_candidates
    t0 = time.time()
    d = art_dir(cfg, split)
    b = cfg["blocking"]
    s1, pool = load_normalized(cfg, split, columns=["entity_id", "name_skel", "name_alt_skel", "name_ns",
                                                    "addr_clean", "country_norm"])
    qi = np.arange(len(s1))  # block every S1; the train sample is selected in stage_features
    q = s1
    tb = token_block(q, pool, k=b["token_top_k"], max_df=b["token_max_df"])
    print(f"[block:{split}] token pass {len(tb):,} pairs [{time.time() - t0:.0f}s]", flush=True)
    parts = [tb]
    emb_path = os.path.join(d, "emb_pool.npy")
    if os.path.exists(emb_path):
        from src.embed import embed_block
        e_s1 = np.load(os.path.join(d, "emb_s1.npy"), mmap_mode="r")
        e_pool = np.load(emb_path, mmap_mode="r")
        eb = embed_block(q, pool, np.ascontiguousarray(e_s1[qi]), e_pool, k=b["embed_top_k"])
        del e_pool
        print(f"[block:{split}] embedding pass {len(eb):,} pairs [{time.time() - t0:.0f}s]", flush=True)
        parts.append(eb)
    else:
        print(f"[block:{split}] no embeddings cached - token pass only", flush=True)
    cands = union_candidates(parts, max_per_s1=b["max_candidates"])
    cands["s1_idx"] = qi[cands["s1_idx"].to_numpy()]
    cands["s1_id"] = s1["entity_id"].to_numpy()[cands["s1_idx"].to_numpy()]
    cands["pool_id"] = pool["entity_id"].to_numpy()[cands["pool_idx"].to_numpy()]
    cands.to_parquet(os.path.join(d, "cands.parquet"), index=False)
    sample = query_index(cfg, split, s1)
    pd.DataFrame({"s1_idx": sample}).to_parquet(os.path.join(d, "query_idx.parquet"), index=False)
    print(f"[block:{split}] {len(cands):,} candidates, {len(cands) / len(qi):.1f} per S1 [{time.time() - t0:.0f}s]")
    if split == "train":
        from src.evaluate import blocking_recall, oracle_f05
        from src.io_utils import load_ground_truth
        gold = load_ground_truth(cfg["paths"]["data_dir"])
        ids = s1["entity_id"].to_numpy()[sample].tolist()
        cs = cands.groupby("s1_id")["pool_id"].apply(list).to_dict()
        print(f"[block:train] recall {blocking_recall(cs, gold, ids):.4f} | oracle F0.5 {oracle_f05(cs, gold, ids):.4f}")


def stage_features(cfg, split):
    """Pair features for all candidates, written in S1 chunks to artefacts/<split>/feats/part_*.parquet."""
    import numpy as np
    from src.features import build_pair_features, global_context
    t0 = time.time()
    d = art_dir(cfg, split)
    fd = os.path.join(d, "feats")
    os.makedirs(fd, exist_ok=True)
    for old in os.listdir(fd):
        os.remove(os.path.join(fd, old))
    cands = pd.read_parquet(os.path.join(d, "cands.parquet"))
    ctx = global_context(cands)  # competition context over ALL S1s, as at test time
    sample = pd.read_parquet(os.path.join(d, "query_idx.parquet"))["s1_idx"].to_numpy()
    if len(sample) < cands["s1_idx"].nunique():
        keep = np.isin(cands["s1_idx"].to_numpy(), sample)
        cands, ctx = cands.loc[keep].reset_index(drop=True), ctx.loc[keep].reset_index(drop=True)
    s1, pool = load_normalized(cfg, split, columns=NORM_COLS)
    uniq = np.unique(cands["s1_idx"].to_numpy())
    step = cfg["train"]["feature_chunk_s1"]
    s1_idx = cands["s1_idx"].to_numpy()
    for n, start in enumerate(range(0, len(uniq), step)):
        lo, hi = uniq[start], uniq[min(start + step, len(uniq)) - 1]
        m = (s1_idx >= lo) & (s1_idx <= hi)
        part = cands.loc[m]
        f = build_pair_features(part, s1, pool).reset_index(drop=True)
        f = pd.concat([f, ctx.loc[m].reset_index(drop=True)], axis=1).astype(np.float32)
        f.insert(0, "s1_id", part["s1_id"].to_numpy())
        f.insert(1, "pool_id", part["pool_id"].to_numpy())
        f.to_parquet(os.path.join(fd, f"part_{n:03d}.parquet"), index=False)
        print(f"[features:{split}] part {n}: {len(f):,} rows, {f.shape[1] - 2} features [{time.time() - t0:.0f}s]", flush=True)


def load_features(cfg, split):
    """Concatenate cached feature parts."""
    fd = os.path.join(art_dir(cfg, split), "feats")
    return pd.concat([pd.read_parquet(os.path.join(fd, p)) for p in sorted(os.listdir(fd))], ignore_index=True)


def stage_rank(cfg, split):
    """train: grouped OOF LightGBM + decision tuning + final model. test: predict probabilities."""
    import json
    from src.ranker import predict_lgbm, train_lgbm
    d = art_dir(cfg, split)
    md = os.path.join(cfg["paths"]["artefacts_dir"], "model")
    os.makedirs(md, exist_ok=True)
    if split == "train":
        from src.decide import tune
        from src.io_utils import load_ground_truth
        gold = load_ground_truth(cfg["paths"]["data_dir"])
        feats = load_features(cfg, "train")
        q = pd.read_parquet(os.path.join(d, "query_idx.parquet"))["s1_idx"].to_numpy()
        all_ids = pd.read_parquet(os.path.join(d, "s1.parquet"), columns=["entity_id"])["entity_id"].to_numpy()
        s1_ids = all_ids[q].tolist()
        y = [int(p in gold.get(s, ())) for s, p in zip(feats["s1_id"], feats["pool_id"])]
        models, oof, feat_cols = train_lgbm(feats, y, cfg)
        df = feats[["s1_id", "pool_id"]].copy()
        df["prob"] = oof
        df.to_parquet(os.path.join(d, "oof.parquet"), index=False)
        params, score = tune(df, gold, s1_ids)
        print(f"[rank:train] stage1 OOF macro F0.5 = {score:.4f} on {len(s1_ids):,} S1", flush=True)
        for i, m in enumerate(models):
            m.save_model(os.path.join(md, f"lgbm_{i}.txt"))
        mc = {"features": feat_cols, "decision": params, "oof_f05": score, "n_models": len(models), "stage2": False}
        # stage 2: add per-S1 probability context from the OOF stage-1 predictions
        from src.ranker import train_stage2
        models2, oof2, feat_cols2 = train_stage2(feats, oof, y, cfg)
        df["prob"] = oof2
        params2, score2 = tune(df, gold, s1_ids)
        print(f"[rank:train] stage2 OOF macro F0.5 = {score2:.4f}", flush=True)
        if score2 > score + 0.0005:
            for i, m in enumerate(models2):
                m.save_model(os.path.join(md, f"lgbm2_{i}.txt"))
            mc.update({"stage2": True, "features2": feat_cols2, "decision": params2, "oof_f05": score2})
            df.to_parquet(os.path.join(d, "oof.parquet"), index=False)
        with open(os.path.join(md, "config.json"), "w") as f:
            json.dump(mc, f, indent=2)
    else:
        with open(os.path.join(md, "config.json")) as f:
            mc = json.load(f)
        from src.ranker import prob_context
        fd = os.path.join(d, "feats")
        out = []
        for p in sorted(os.listdir(fd)):
            feats = pd.read_parquet(os.path.join(fd, p))  # parts hold whole S1 groups
            df = feats[["s1_id", "pool_id"]].copy()
            df["prob"] = predict_lgbm(feats, md, mc)
            if mc.get("stage2"):
                f2 = pd.concat([feats.reset_index(drop=True), prob_context(feats, df["prob"].to_numpy())], axis=1)
                df["prob"] = predict_lgbm(f2, md, mc, stage=2)
            out.append(df)
            print(f"[rank:test] {p} scored", flush=True)
        pd.concat(out, ignore_index=True).to_parquet(os.path.join(d, "probs.parquet"), index=False)


def stage_decide(cfg, split):
    """test: probabilities -> matches; write matching_results.tsv + candidate_pairs.tsv and validate."""
    import json
    import subprocess
    import sys
    from src.decide import decide
    from src.io_utils import write_tsv
    if split != "test":
        print("[decide] train decisions are tuned and scored inside --stage rank")
        return
    d = art_dir(cfg, "test")
    md = os.path.join(cfg["paths"]["artefacts_dir"], "model")
    with open(os.path.join(md, "config.json")) as f:
        params = json.load(f)["decision"]
    s1_ids = pd.read_parquet(os.path.join(d, "s1.parquet"), columns=["entity_id"])["entity_id"].tolist()
    probs = pd.read_parquet(os.path.join(d, "probs.parquet"))
    matches = decide(probs, s1_ids, params)
    cands = pd.read_parquet(os.path.join(d, "cands.parquet"), columns=["s1_id", "pool_id"])
    cand_lists = cands.groupby("s1_id")["pool_id"].apply(list).to_dict()
    out = cfg["paths"]["output_dir"]
    write_tsv(s1_ids, matches, os.path.join(out, "matching_results.tsv"), "matched_entity_ids")
    write_tsv(s1_ids, cand_lists, os.path.join(out, "candidate_pairs.tsv"), "candidate_entity_ids")
    n_empty = sum(1 for s in s1_ids if not matches.get(s))
    print(f"[export] {len(s1_ids):,} S1 rows, {n_empty / len(s1_ids):.1%} empty, "
          f"{sum(len(v) for v in matches.values()) / len(s1_ids):.2f} matches per S1", flush=True)
    subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", os.path.join(out, "matching_results.tsv"),
                    "--candidate", os.path.join(out, "candidate_pairs.tsv"), "--test-dir",
                    os.path.join(cfg["paths"]["data_dir"], "test")])


def main():
    """Run the requested pipeline stage(s) for the requested split."""
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    stages = STAGES if args.stage == "all" else [args.stage]
    for st in stages:
        if st == "normalize":
            stage_normalize(cfg, args.split)
        elif st == "embed":
            stage_embed(cfg, args.split)
        elif st == "block":
            stage_block(cfg, args.split)
        elif st == "features":
            stage_features(cfg, args.split)
        elif st == "rank":
            stage_rank(cfg, args.split)
        elif st == "decide":
            stage_decide(cfg, args.split)
        elif st == "cross_encoder":
            print("[cross_encoder] not used yet - skipped")
        else:
            raise NotImplementedError(f"stage={st}")


if __name__ == "__main__":
    main()
