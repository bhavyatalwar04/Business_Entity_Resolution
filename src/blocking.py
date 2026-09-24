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


def _record_tokens(df):
    """Token list per record: 'n:' name-skeleton tokens (+ alias), 'a:' address tokens."""
    out = []
    for skel, alt_skel, ns, addr in zip(df["name_skel"], df["name_alt_skel"], df["name_ns"], df["addr_clean"]):
        toks = {"n:" + t for t in skel.split() if len(t) > 1}
        toks.update("n:" + t for t in alt_skel.split() if len(t) > 1)
        if len(ns) >= 6:
            toks.add("s:" + ns)  # whole name without spaces: catches 'urologystrategichealth'
        toks.update("a:" + t for t in addr.split())
        out.append(list(toks))
    return out


def _tfidf_matrix(tok_lists, vocab):
    """Binary token matrix restricted to `vocab` (token -> column)."""
    indptr, indices = [0], []
    for toks in tok_lists:
        cols = [vocab[t] for t in toks if t in vocab]
        indices.extend(cols)
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    return sp.csr_matrix((data, np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int64)),
                         shape=(len(tok_lists), len(vocab)))


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
    make the sparse product dense and carry little identity signal. Scores are cosine-like
    (IDF-weighted overlap normalised by both records' IDF mass)."""
    frames = []
    s1_tok, pool_tok = _record_tokens(s1), _record_tokens(pool)
    s1_c, pool_c = s1["country_norm"].to_numpy(), pool["country_norm"].to_numpy()
    for country in pd.unique(s1_c):
        si = np.flatnonzero(s1_c == country)
        pi = np.flatnonzero(pool_c == country)
        if len(pi) == 0:
            pi = np.arange(len(pool))  # unseen label on the pool side: block against everything
        df_count = {}
        for i in pi:
            for t in pool_tok[i]:
                df_count[t] = df_count.get(t, 0) + 1
        vocab = {t: j for j, t in enumerate(t for t, c in df_count.items() if c <= max_df)}
        idf = np.zeros(len(vocab), dtype=np.float32)
        for t, j in vocab.items():
            idf[j] = np.log(1 + len(pi) / df_count[t])
        B = _tfidf_matrix([pool_tok[i] for i in pi], vocab)
        A = _tfidf_matrix([s1_tok[i] for i in si], vocab)
        W = sp.diags(idf)
        A, B = (A @ W).tocsr(), (B @ W).tocsr()
        a_norm = np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel()) + 1e-6
        b_norm = np.sqrt(np.asarray(B.multiply(B).sum(1)).ravel()) + 1e-6
        A = sp.diags(1 / a_norm) @ A
        B = (sp.diags(1 / b_norm) @ B).T.tocsr()
        # weight by idf again on one side so a shared rare token counts more than a common one
        for start in range(0, A.shape[0], chunk):
            S = (A[start:start + chunk] @ B).tocsr()
            r, c, v = _topk_rows(S, k)
            frames.append(pd.DataFrame({"s1_idx": si[start + r], "pool_idx": pi[c], "tok_score": v}))
            if verbose:
                print(f"    token_block[{country}] {start + A[start:start + chunk].shape[0]}/{A.shape[0]}", flush=True)
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
