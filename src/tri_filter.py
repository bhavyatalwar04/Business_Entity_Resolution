"""Transitive-consistency filter for multi-source matches (TransClean-style, arXiv 2506.04006).

If an S1 is matched to S2-record x AND S3-record y, then x and y describe the same business, so a cross-encoder
should score the pair (x, y) high. When it scores (x, y) below t, one of the two S1 matches is likely a false positive:
drop the one with the lower stacker probability. Conflicts are resolved weakest-evidence first (lowest p(x, y)), and a
record already dropped is not used again. Pairs with no (x, y) score are left alone (no evidence, no action).

  eval  : tune t on labelled fold-0 (India/US) with the same cross-fitted halves as blend_eval, report held-out F0.5
          with vs without the filter. PASS = with >= without.
  apply : filter an existing matching_results.tsv, ONLY for S1 of the given countries (default: unseen = France).

Inputs
  stacker probs : parquet (s1_id, pool_id, <prob col>)         e.g. artefacts/exp/stack_oof_<tag>.parquet (eval)
  tri scores    : parquet (s1_id = S2 id, pool_id = S3 id, ce_prob)   handoff/ce_tri_out/tri_fold0* / tri_fr*

Usage
  python -m src.tri_filter eval  --stack artefacts/exp/stack_oof_v11.parquet --col stack_x_all \
                                 --tri "handoff/ce_tri_out/tri_fold0_part*.parquet"
  python -m src.tri_filter apply --tsv output/v11_qwen/matching_results.tsv --probs <test probs parquet> \
                                 --tri "handoff/ce_tri_out/tri_fr_part*.parquet" --t 0.05 --out output/v15_tri
"""
import argparse
import glob
import os
import zlib

import numpy as np
import pandas as pd

from src.decide import FAST_GRID, decide, tune
from src.evaluate import f05
from src.io_utils import load_ground_truth, read_tsv, write_tsv

T_GRID = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def read_glob(pattern, columns=None):
    return pd.concat([pd.read_parquet(f, columns=columns) for f in sorted(glob.glob(pattern))], ignore_index=True)


def tri_lookup(tri):
    """{(s2_id, s3_id): prob} for S2 x S3 pairs (either orientation in the input)."""
    a, b, p = tri["s1_id"].to_numpy(), tri["pool_id"].to_numpy(), tri["ce_prob"].to_numpy()
    out = {}
    for x, y, v in zip(a, b, p):
        k = (x, y) if x.startswith("S2") else (y, x)
        out[k] = max(out.get(k, 0.0), float(v))
    return out


def filter_sets(pred, prob, tri, t):
    """pred {s1: [ids]}, prob {(s1, id): stacker prob}, tri {(s2, s3): p} -> (new pred, n dropped)."""
    out, dropped = {}, 0
    for s1, ids in pred.items():
        s2 = [i for i in ids if i.startswith("S2")]
        s3 = [i for i in ids if i.startswith("S3")]
        if not s2 or not s3:
            out[s1] = ids
            continue
        conflicts = sorted((tri[(x, y)], x, y) for x in s2 for y in s3 if (x, y) in tri and tri[(x, y)] < t)
        gone = set()
        for _, x, y in conflicts:
            if x in gone or y in gone:
                continue
            gone.add(x if prob.get((s1, x), 0.0) <= prob.get((s1, y), 0.0) else y)
        dropped += len(gone)
        out[s1] = [i for i in ids if i not in gone]
    return out, dropped


def macro(pred, gold, ids):
    return float(np.mean([f05(pred.get(s, []), gold.get(s, ())) for s in ids]))


def cmd_eval(args):
    gold = load_ground_truth("dataset")
    st = pd.read_parquet(args.stack)
    df = st[["s1_id", "pool_id", args.col]].rename(columns={args.col: "prob"})
    ids = sorted(df["s1_id"].unique())
    s1 = read_tsv("dataset/train/train_source1.tsv", usecols=["entity_id", "country"])
    country = dict(zip(s1["entity_id"], s1["country"].str.lower()))
    hh = {s: zlib.crc32(s.encode()) % 10 for s in ids}
    halves = [[s for s in ids if hh[s] == 0], [s for s in ids if hh[s] == 5]]
    tri = tri_lookup(read_glob(args.tri, ["s1_id", "pool_id", "ce_prob"]))
    prob = dict(zip(zip(df["s1_id"], df["pool_id"]), df["prob"]))
    print(f"fold-0 S1 {len(ids):,} | tri pairs {len(tri):,}", flush=True)
    base, filt, chosen = {}, {}, []
    for a, b in ((0, 1), (1, 0)):
        da = df[df["s1_id"].isin(set(halves[a]))]
        db = df[df["s1_id"].isin(set(halves[b]))]
        params, _ = tune(da, gold, halves[a], verbose=False, grid=FAST_GRID, o2o_opts=(True,))
        pa, pb = decide(da, halves[a], params), decide(db, halves[b], params)
        best_t, best_s = None, macro(pa, gold, halves[a])  # t must beat 'no filter' on the tuning half
        for t in T_GRID:
            s = macro(filter_sets(pa, prob, tri, t)[0], gold, halves[a])
            if s > best_s:
                best_t, best_s = t, s
        chosen.append(best_t)
        base.update(pb)
        filt.update(pb if best_t is None else filter_sets(pb, prob, tri, best_t)[0])
    all_ids = halves[0] + halves[1]
    for name, pred in (("without filter", base), ("with filter", filt)):
        by_c = pd.Series({s: f05(pred.get(s, []), gold.get(s, ())) for s in all_ids}).groupby(
            pd.Series({s: country.get(s) for s in all_ids})).mean()
        print(f"  {name:15s} held-out macro F0.5 {macro(pred, gold, all_ids):.4f} | "
              + " | ".join(f"{c} {v:.4f}" for c, v in by_c.items()), flush=True)
    print(f"  t chosen per half: {chosen}")
    for t in T_GRID:  # how much the filter touches, for reading the France application
        _, n = filter_sets(base, prob, tri, t)
        print(f"    t={t:<6} drops {n:,} matches on fold-0 ({n / max(sum(len(v) for v in base.values()), 1):.2%})")


def cmd_apply(args):
    sub = read_tsv(args.tsv)
    col = [c for c in sub.columns if c != "source1_entity_id"][0]
    pred = {s: [x for x in str(m).split(",") if x] for s, m in zip(sub["source1_entity_id"], sub[col].fillna(""))}
    s1 = read_tsv("dataset/test/test_source1.tsv", usecols=["entity_id", "country"])
    seen = set(read_tsv("dataset/train/train_source1.tsv", usecols=["country"])["country"].unique())
    target = set(s1.loc[s1["country"].isin(args.countries.split(",")) if args.countries
                        else ~s1["country"].isin(seen), "entity_id"])
    pr = pd.read_parquet(args.probs) if args.probs.endswith(".parquet") else read_glob(args.probs)
    pcol = [c for c in pr.columns if c not in ("s1_id", "pool_id")][0]
    pr = pr[pr["s1_id"].isin(target)]
    prob = dict(zip(zip(pr["s1_id"], pr["pool_id"]), pr[pcol]))
    tri = tri_lookup(read_glob(args.tri, ["s1_id", "pool_id", "ce_prob"]))
    sub_t = {s: v for s, v in pred.items() if s in target}
    new_t, n = filter_sets(sub_t, prob, tri, args.t)
    pred.update(new_t)
    before, after = sum(len(v) for v in sub_t.values()), sum(len(v) for v in new_t.values())
    print(f"target S1 {len(target):,} | matches {before:,} -> {after:,} (dropped {n:,}); other S1 unchanged")
    os.makedirs(args.out, exist_ok=True)
    ids = list(sub["source1_entity_id"])
    write_tsv(ids, [pred[s] for s in ids], os.path.join(args.out, "matching_results.tsv"), col)
    print(f"wrote {os.path.join(args.out, 'matching_results.tsv')}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("eval")
    e.add_argument("--stack", required=True)
    e.add_argument("--col", required=True)
    e.add_argument("--tri", required=True)
    a = sp.add_parser("apply")
    a.add_argument("--tsv", required=True)
    a.add_argument("--probs", required=True, help="parquet/glob with s1_id, pool_id, prob of the test pairs")
    a.add_argument("--tri", required=True)
    a.add_argument("--t", type=float, required=True)
    a.add_argument("--countries", default="", help="comma list; default = countries unseen in train (France)")
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    cmd_eval(args) if args.cmd == "eval" else cmd_apply(args)


if __name__ == "__main__":
    main()
