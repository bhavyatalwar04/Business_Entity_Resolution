"""Candidate generation.

Every true pair in training shares the same country label, so blocking runs inside each
country value (an open set of strings - France is just another value; records whose
country appears only on one side still get blocked against the whole pool).

Passes (union):
  token_block : IDF-weighted overlap of rare tokens (name skeleton tokens + address tokens).
  embed_block : multilingual sentence-embedding kNN on 'name | address' (GPU, see embed.py).

Output: DataFrame(s1_idx, pool_idx, tok_score, tok_rank[, emb_score, emb_rank]).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _prefixed(col, prefix):
    """'ab cd' -> 'P:ab P:cd' for a whole string column (empty stays empty)."""
    col = col.fillna("")
    out = prefix + col.str.replace(" ", " " + prefix, regex=False)
    return out.where(col.str.len() > 0, "")


def token_docs(df):
    """One space-separated token document per record:
    'n:' name-skeleton tokens (+ alias), 's:' whole no-space name, 'a:' address tokens."""
    ns = df["name_ns"].fillna("")
    s_tok = ("s:" + ns).where(ns.str.len() >= 6, "")
    return (_prefixed(df["name_skel"], "n:") + " " + _prefixed(df["name_alt_skel"], "n:") + " "
            + s_tok + " " + _prefixed(df["addr_clean"], "a:")).tolist()


def _topk_rows(S, k):
    """Top-k columns per row of a CSR score matrix. Returns (rows, cols, scores)."""
    rows, cols, vals = [], [], []
    indptr, indices, data = S.indptr, S.indices, S.data
    for r in range(S.shape[0]):
        a, b = indptr[r], indptr[r + 1]
        if a == b:
            continue
        d = data[a:b]
        if b - a > k:
            sel = np.argpartition(-d, k - 1)[:k]
        else:
            sel = np.arange(b - a)
        rows.append(np.full(len(sel), r, dtype=np.int64))
        cols.append(indices[a:b][sel])
        vals.append(d[sel])
    if not rows:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.float32)
    return np.concatenate(rows), np.concatenate(cols).astype(np.int64), np.concatenate(vals)


def token_block(s1, pool, k=30, max_df=3000, chunk=20000, verbose=False):
    """Pass C: IDF-weighted rare-token overlap within each country; top-k pool records per S1.

    Tokens with document frequency above `max_df` (in the country's pool) are ignored - they
    make the sparse product dense and carry little identity signal. Rows are IDF-weighted and
    L2-normalised, so the score is a cosine over rare tokens."""
    from sklearn.feature_extraction.text import CountVectorizer
    frames = []
    s1_c, pool_c = s1["country_norm"].to_numpy(), pool["country_norm"].to_numpy()
    for country in pd.unique(s1_c):
        si = np.flatnonzero(s1_c == country)
        pi = np.flatnonzero(pool_c == country)
        if len(pi) == 0:
            pi = np.arange(len(pool))  # label unseen on the pool side: block against everything
        vec = CountVectorizer(analyzer=str.split, binary=True, dtype=np.float32)
        B = vec.fit_transform(token_docs(pool.iloc[pi])).tocsc()
        df_count = np.diff(B.indptr)
        keep = np.flatnonzero((df_count <= max_df) & (df_count > 0))
        idf = np.log1p(len(pi) / df_count[keep]).astype(np.float32)
        B = (B[:, keep] @ sp.diags(idf)).tocsr()
        A = vec.transform(token_docs(s1.iloc[si])).tocsc()[:, keep]
        A = (A @ sp.diags(idf)).tocsr()
        A = sp.diags(1 / (np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel()) + 1e-6)) @ A
        B = sp.diags(1 / (np.sqrt(np.asarray(B.multiply(B).sum(1)).ravel()) + 1e-6)) @ B
        BT = B.T.tocsr().astype(np.float32)
        A = A.tocsr().astype(np.float32)
        del B
        for start in range(0, A.shape[0], chunk):
            S = (A[start:start + chunk] @ BT).tocsr()
            r, c, v = _topk_rows(S, k)
            frames.append(pd.DataFrame({"s1_idx": si[start + r], "pool_idx": pi[c], "tok_score": v}))
            if verbose:
                print(f"    token_block[{country}] {min(start + chunk, A.shape[0]):,}/{A.shape[0]:,}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out["tok_rank"] = out.groupby("s1_idx")["tok_score"].rank(ascending=False, method="first").astype(np.int16)
    return out


def union_candidates(parts, max_per_s1=None):
    """Outer-join blocking passes on (s1_idx, pool_idx); optional cap by best rank across passes."""
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=["s1_idx", "pool_idx"], how="outer")
    rank_cols = [c for c in out.columns if c.endswith("_rank")]
    for c in rank_cols:
        out[c] = out[c].fillna(999).astype(np.int16)
    for c in [c for c in out.columns if c.endswith("_score")]:
        out[c] = out[c].fillna(0).astype(np.float32)
    out["best_rank"] = out[rank_cols].min(axis=1)
    if max_per_s1:
        out = out.sort_values(["s1_idx", "best_rank"]).groupby("s1_idx").head(max_per_s1)
    return out.sort_values(["s1_idx", "pool_idx"]).reset_index(drop=True)
