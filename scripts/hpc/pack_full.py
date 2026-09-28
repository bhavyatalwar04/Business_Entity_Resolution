"""Pack step-3 outputs into < 90 MB pieces under handoff/full_out/ (branch full-results)."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.evaluate import fold_of

OUT, LIM, ROWS = "handoff/full_out", 90 * 2**20, 4_000_000
os.makedirs(OUT, exist_ok=True)


def write_parts(table_iter, name):
    n = 0
    for i, t in enumerate(table_iter):
        p = f"{OUT}/{name}_part{i:02d}.parquet"
        pq.write_table(t, p, compression="zstd")
        assert os.path.getsize(p) < LIM, p
        n += t.num_rows
    print(f"{name}: {i + 1} parts, {n:,} rows", flush=True)


ONLY_OOF = len(sys.argv) > 1 and sys.argv[1] == "oof"

for f in ([] if ONLY_OOF else ["matching_results", "candidate_pairs"]):
    src = f"output/full/{f}.tsv"
    subprocess.run(f"gzip -c {src} > {OUT}/{f}.tsv.gz", shell=True, check=True)
    if os.path.getsize(f"{OUT}/{f}.tsv.gz") >= LIM:
        subprocess.run(f"split -b 85m -d {OUT}/{f}.tsv.gz {OUT}/{f}.tsv.gz.part && rm {OUT}/{f}.tsv.gz", shell=True, check=True)

if not ONLY_OOF:
    pf = pq.ParquetFile("artefacts/test/probs_model_full.parquet")
    write_parts((pa.Table.from_batches([b]) for b in pf.iter_batches(batch_size=ROWS)), "probs_model_full")

d = "artefacts/train"
s1 = pd.read_parquet(f"{d}/s1.parquet", columns=["entity_id"])["entity_id"].to_numpy()
pool = np.concatenate([pd.read_parquet(f"{d}/s{k}.parquet", columns=["entity_id"])["entity_id"].to_numpy() for k in (2, 3)])
fold1 = {}


def oof_tables():
    for b in pq.ParquetFile(f"{d}/oof.parquet").iter_batches(batch_size=ROWS, columns=["s1_id", "pool_id", "prob"]):
        t = b.to_pandas()
        s = s1[t["s1_id"].to_numpy()]
        fo = np.fromiter((fold1.setdefault(x, fold_of(x, 5)) for x in s), np.int8, len(s))
        yield pa.Table.from_pandas(pd.DataFrame({"s1_id": s, "pool_id": pool[t["pool_id"].to_numpy()],
                                                 "prob": t["prob"].astype(np.float32), "fold": fo}), preserve_index=False)


write_parts(oof_tables(), "oof_train_full")
print("PACK DONE", flush=True)
