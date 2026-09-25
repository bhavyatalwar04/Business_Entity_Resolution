"""Rebuild matching_results.tsv and candidate_pairs.tsv (in output/v1/) from the committed .gz files."""
import glob
import gzip
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))


def gunzip(src_parts, dst):
    """Concatenate gzip parts in order and decompress them to dst."""
    joined = dst + ".gz.tmp"
    with open(joined, "wb") as out:
        for p in src_parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out)
    with gzip.open(joined, "rb") as f, open(dst, "wb") as out:
        shutil.copyfileobj(f, out)
    os.remove(joined)
    print("wrote", dst)


if __name__ == "__main__":
    gunzip([os.path.join(HERE, "matching_results.tsv.gz")], os.path.join(HERE, "matching_results.tsv"))
    gunzip(sorted(glob.glob(os.path.join(HERE, "candidate_pairs.tsv.gz.part*"))), os.path.join(HERE, "candidate_pairs.tsv"))
