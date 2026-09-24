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

STAGES = ["normalize", "embed", "block", "features", "cross_encoder", "rank", "decide", "export"]

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
        t0 = time.time()
        emb = encode(record_text(df), model_name=model)
        np.save(os.path.join(d, f"emb_{name}.npy"), emb)
        print(f"[embed:{split}] {name} {emb.shape} [{time.time() - t0:.0f}s]", flush=True)


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
        else:
            raise NotImplementedError(f"stage={st}")


if __name__ == "__main__":
    main()
