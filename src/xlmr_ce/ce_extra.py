"""Cross-encoder scores for step-3 candidate pairs the handoff pair files do not cover.

extra test  : probs_model_full rows with prob >= 0.001 not in handoff/ce/test_pairs_*
extra fold0 : step-3 OOF rows with fold == 0 and prob >= 0.001 not in the fold-0 rows of handoff/ce/train_pairs_*
Each list is scored with every checkpoint given (--ckpts tag=dir:out_dir ...) and written as
ce_extra_{test,fold0}_part*.parquet (s1_id, pool_id, ce_prob) into that out_dir.

Usage: PYTHONPATH=. python -m src.xlmr_ce.ce_extra --ckpts base=artefacts/ce_base/final:handoff/ce_out large=artefacts/ce_large/final:handoff/ce_large_out
"""
import argparse
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.xlmr_ce.ce_rescore import attach_text, load_texts, log, predict, read_pairs, write_parts
from src.evaluate import fold_of

THR = 0.001


def to_str_ids(df, split):
    """Integer s1/pool indices (as stored by rank:train) -> entity id strings."""
    if not pd.api.types.is_integer_dtype(df["s1_id"]):
        return df
    d = f"artefacts/{split}"
    s1 = pd.read_parquet(f"{d}/s1.parquet", columns=["entity_id"])["entity_id"].to_numpy()
    pool = np.concatenate([pd.read_parquet(f"{d}/s{k}.parquet", columns=["entity_id"])["entity_id"].to_numpy() for k in (2, 3)])
    return df.assign(s1_id=s1[df["s1_id"].to_numpy()], pool_id=pool[df["pool_id"].to_numpy()])


def anti_join(new, old):
    key = ["s1_id", "pool_id"]
    m = new[key].merge(old[key].drop_duplicates(), on=key, how="left", indicator=True)
    return m.loc[m["_merge"] == "left_only", key].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    args = ap.parse_args()

    oof = to_str_ids(pd.read_parquet("artefacts/train/oof.parquet", columns=["s1_id", "pool_id", "prob"]), "train")
    oof = oof[oof["prob"] >= THR]
    oof = oof[np.fromiter((fold_of(s, 5) == 0 for s in oof["s1_id"]), bool, len(oof))]
    old0 = read_pairs("handoff/ce/train_pairs_part*.parquet", columns=["s1_id", "pool_id", "fold"])
    extra0 = anti_join(oof, old0[old0["fold"] == 0])
    log(f"fold-0 step-3 pairs >= {THR}: {len(oof):,} over {oof['s1_id'].nunique():,} S1 -> extra {len(extra0):,}")
    del oof, old0

    te = to_str_ids(pd.read_parquet("artefacts/test/probs_model_full.parquet", columns=["s1_id", "pool_id", "prob"]), "test")
    te = te[te["prob"] >= THR]
    extrat = anti_join(te, read_pairs("handoff/ce/test_pairs_part*.parquet", columns=["s1_id", "pool_id"]))
    log(f"test step-3 pairs >= {THR}: {len(te):,} -> extra {len(extrat):,}")
    del te

    texts = {"train": load_texts("dataset", "train"), "test": load_texts("dataset", "test")}
    lists = {"fold0": (extra0, "train"), "test": (extrat, "test")}
    t = {k: attach_text(df, texts[split]) for k, (df, split) in lists.items()}
    del texts
    opts = SimpleNamespace(infer_bs=1024, max_len=128)
    for spec in args.ckpts:
        tag, rest = spec.split("=", 1)
        ckpt, out = rest.split(":", 1)
        if not os.path.exists(os.path.join(ckpt, "config.json")):
            log(f"[{tag}] no checkpoint at {ckpt} - skipped")
            continue
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt).cuda()
        os.makedirs(out, exist_ok=True)
        for name, (df, _) in lists.items():
            res = df.copy()
            res["ce_prob"] = predict(model, tok, *t[name], opts) if len(df) else np.empty(0, np.float32)
            write_parts(res, out, f"ce_extra_{name}")
        open(os.path.join(out, "ce_extra.DONE"), "w").close()
        log(f"[{tag}] done -> {out}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
