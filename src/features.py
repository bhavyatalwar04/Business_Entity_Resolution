"""Pair features for LightGBM.

Groups:
  name     : fuzzy ratios on core / clean / skeleton / no-space / alias views
  address  : fuzzy ratios, token Jaccard, house-number / postal / landmark agreement
  blocking : token-pass and embedding-pass scores and ranks
  context  : rank and gap of this candidate among its S1's candidates, and of the S1 among
             the candidate's S1s (how contested the candidate is)
  meta     : source (2/3), lengths, missing flags. Country is never one-hot encoded.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist


def _sim(a, b, scorer):
    """Element-wise similarity of aligned string lists scaled to [0, 1]."""
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / 100.0


def _set_stats(sa, sb):
    """Jaccard, intersection size, both-present flag for aligned lists of space-joined token strings."""
    n = len(sa)
    jac = np.zeros(n, np.float32)
    inter = np.zeros(n, np.float32)
    both = np.zeros(n, np.int8)
    for i, (x, y) in enumerate(zip(sa, sb)):
        if x and y:
            xs, ys = set(x.split()), set(y.split())
            both[i] = 1
            k = len(xs & ys)
            inter[i] = k
            jac[i] = k / len(xs | ys)
    return jac, inter, both


def _first_num(addr):
    """First digit run of an address string (house / door number), or ''."""
    for t in addr.split():
        if t.isdigit():
            return t
    return ""


def add_group_context(df, key, col, prefix):
    """Rank (1 = best) and gap-to-best of `col` within groups of `key` (in place)."""
    g = df.groupby(key, sort=False)[col]
    df[f"{prefix}_rank_{col}"] = g.rank(ascending=False, method="min").astype(np.float32)
    df[f"{prefix}_gap_{col}"] = (g.transform("max") - df[col]).astype(np.float32)


def global_context(cands):
    """Context features that need the whole candidate table (reverse side = per pool record)."""
    out = pd.DataFrame(index=cands.index)
    out["n_cands_s1"] = cands.groupby("s1_idx")["pool_idx"].transform("size").astype(np.float32)
    out["n_s1_per_cand"] = cands.groupby("pool_idx")["s1_idx"].transform("size").astype(np.float32)
    tmp = cands[["s1_idx", "pool_idx"]].copy()
    for col in [c for c in ("emb_score", "tok_score") if c in cands.columns]:
        tmp[col] = cands[col]
        add_group_context(tmp, "pool_idx", col, "cand")
        out[f"cand_rank_{col}"] = tmp[f"cand_rank_{col}"]
        out[f"cand_gap_{col}"] = tmp[f"cand_gap_{col}"]
    return out


def build_pair_features(pairs, s1, pool):
    """Candidate pairs (with blocking columns) -> numeric feature DataFrame aligned with `pairs`."""
    ia = pairs["s1_idx"].to_numpy()
    ib = pairs["pool_idx"].to_numpy()
    A = s1.iloc[ia].reset_index(drop=True)
    B = pool.iloc[ib].reset_index(drop=True)
    f = pd.DataFrame(index=pairs.index)

    ac, bc = A["name_core"].tolist(), B["name_core"].tolist()
    f["n_ratio"] = _sim(ac, bc, fuzz.ratio)
    f["n_partial"] = _sim(ac, bc, fuzz.partial_ratio)
    f["n_tsort"] = _sim(ac, bc, fuzz.token_sort_ratio)
    f["n_tset"] = _sim(ac, bc, fuzz.token_set_ratio)
    f["n_jw"] = cpdist(ac, bc, scorer=JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32)
    f["n_clean_ratio"] = _sim(A["name_clean"].tolist(), B["name_clean"].tolist(), fuzz.ratio)
    ask, bsk = A["name_skel"].tolist(), B["name_skel"].tolist()
    f["n_skel_ratio"] = _sim(ask, bsk, fuzz.ratio)
    f["n_skel_tset"] = _sim(ask, bsk, fuzz.token_set_ratio)
    f["n_skel_jac"], f["n_skel_inter"], _ = _set_stats(ask, bsk)
    ans, bns = A["name_ns"].tolist(), B["name_ns"].tolist()
    f["n_ns_ratio"] = _sim(ans, bns, fuzz.ratio)
    f["n_ns_partial"] = _sim(ans, bns, fuzz.partial_ratio)
    f["n_exact"] = (A["name_core"].values == B["name_core"].values).astype(np.int8)
    f["n_ns_eq"] = (A["name_ns"].values == B["name_ns"].values).astype(np.int8)
    b_alt = B["name_alt"].tolist()
    has_alt = np.array([bool(x) for x in b_alt])
    alt_sim = _sim(ac, [x or "\x00" for x in b_alt], fuzz.token_set_ratio)
    f["n_alt_tset"] = np.where(has_alt, alt_sim, -1).astype(np.float32)
    f["b_has_alt"] = has_alt.astype(np.int8)

    aa, ba = A["addr_clean"].tolist(), B["addr_clean"].tolist()
    f["a_ratio"] = _sim(aa, ba, fuzz.ratio)
    f["a_partial"] = _sim(aa, ba, fuzz.partial_ratio)
    f["a_tset"] = _sim(aa, ba, fuzz.token_set_ratio)
    f["a_tsort"] = _sim(aa, ba, fuzz.token_sort_ratio)
    f["a_jac"], f["a_inter"], _ = _set_stats(aa, ba)
    f["b_addr_empty"] = (B["addr_clean"].str.len().values == 0).astype(np.int8)
    f["num_jac"], f["num_inter"], f["num_both"] = _set_stats(A["nums"].tolist(), B["nums"].tolist())
    f["num_conflict"] = ((f["num_both"] == 1) & (f["num_inter"] == 0)).astype(np.int8)
    fa = np.array([_first_num(x) for x in aa], dtype=object)
    fb = np.array([_first_num(x) for x in ba], dtype=object)
    f["house_eq"] = np.where((fa != "") & (fb != ""), (fa == fb).astype(np.int8), -1).astype(np.int8)
    f["pc_jac"], _, f["pc_both"] = _set_stats(A["postal"].tolist(), B["postal"].tolist())
    f["lm_jac"], f["lm_inter"], _ = _set_stats(A["landmarks"].tolist(), B["landmarks"].tolist())

    f["source"] = B["source"].values.astype(np.int8)
    f["len_name_a"] = A["name_core"].str.len().values.astype(np.float32)
    f["len_name_b"] = B["name_core"].str.len().values.astype(np.float32)
    f["len_addr_a"] = A["addr_clean"].str.len().values.astype(np.float32)
    f["len_addr_b"] = B["addr_clean"].str.len().values.astype(np.float32)

    for c in ("tok_score", "tok_rank", "emb_score", "emb_rank", "best_rank"):
        if c in pairs.columns:
            f[c] = pairs[c].to_numpy().astype(np.float32)

    # S1-side context (pairs are chunked by S1, so each S1's candidates are all present)
    f["_s1"] = ia
    f["name_addr"] = 0.6 * f["n_tset"] + 0.4 * f["a_tset"]
    for col in ["name_addr", "n_tset", "a_tset", "n_skel_ratio"] + [c for c in ("emb_score", "tok_score") if c in f]:
        add_group_context(f, "_s1", col, "s1")
    return f.drop(columns=["_s1"])


def features_in_chunks(cands, s1, pool, s1_chunk=100_000, verbose=True):
    """Build features for all candidates, chunked by S1 index to bound memory; adds global context."""
    ctx = global_context(cands)
    parts = []
    s1_ids = cands["s1_idx"].to_numpy()
    uniq = np.unique(s1_ids)
    for start in range(0, len(uniq), s1_chunk):
        lo, hi = uniq[start], uniq[min(start + s1_chunk, len(uniq)) - 1]
        mask = (s1_ids >= lo) & (s1_ids <= hi)
        parts.append(build_pair_features(cands.loc[mask], s1, pool).set_index(cands.index[mask]))
        if verbose:
            print(f"    features {min(start + s1_chunk, len(uniq)):,}/{len(uniq):,} S1", flush=True)
    feats = pd.concat(parts).loc[cands.index]
    return pd.concat([feats, ctx], axis=1)
