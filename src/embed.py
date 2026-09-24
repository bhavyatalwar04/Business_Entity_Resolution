"""Dense embedding blocking on GPU (Pass B).

Model: intfloat/multilingual-e5-small (MIT, 118M params) - optionally fine-tuned on training
pairs (see finetune_embed.py). Records are encoded as 'name | address' and compared with
exact inner-product kNN on the GPU, chunked over both queries and pool so it fits in 8 GB.
"""
import numpy as np
import pandas as pd
import torch


def record_text(df):
    """Text fed to the encoder for each record: cleaned core name | cleaned address."""
    return ("query: " + df["name_clean"] + " | " + df["addr_clean"]).tolist()


def encode(texts, model_name="intfloat/multilingual-e5-small", batch_size=512, device="cuda", max_len=64):
    """Encode texts to L2-normalised float16 vectors (numpy). Texts are length-sorted for speed."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_len
    model.half()
    order = np.argsort([len(t) for t in texts])
    out = np.empty((len(texts), model.get_sentence_embedding_dimension()), dtype=np.float16)
    step = 200_000
    for s in range(0, len(texts), step):
        idx = order[s:s + step]
        emb = model.encode([texts[i] for i in idx], batch_size=batch_size, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
        out[idx] = emb.astype(np.float16)
        print(f"    encoded {min(s + step, len(texts)):,}/{len(texts):,}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def knn(queries, base, k=30, q_chunk=2048, b_chunk=1_000_000, device="cuda"):
    """Exact top-k inner-product neighbours of each query row in `base` (both float16 numpy).
    Returns (idx int64 [nq, k], sim float32 [nq, k])."""
    nq = len(queries)
    k = min(k, len(base))
    best_s = torch.full((nq, k), -2.0, dtype=torch.float32)
    best_i = torch.zeros((nq, k), dtype=torch.int64)
    for bs in range(0, len(base), b_chunk):
        B = torch.from_numpy(base[bs:bs + b_chunk]).to(device)
        for qs in range(0, nq, q_chunk):
            Q = torch.from_numpy(queries[qs:qs + q_chunk]).to(device)
            S = Q @ B.T
            s, i = torch.topk(S, min(k, S.shape[1]), dim=1)
            s, i = s.float().cpu(), (i + bs).cpu()
            cs = torch.cat([best_s[qs:qs + q_chunk], s], 1)
            ci = torch.cat([best_i[qs:qs + q_chunk], i], 1)
            top = torch.topk(cs, k, dim=1)
            best_s[qs:qs + q_chunk] = top.values
            best_i[qs:qs + q_chunk] = torch.gather(ci, 1, top.indices)
        del B
        torch.cuda.empty_cache()
    return best_i.numpy(), best_s.numpy()


def embed_block(s1, pool, s1_emb, pool_emb, k=30):
    """Pass B: per-country kNN on embeddings; returns DataFrame(s1_idx, pool_idx, emb_score, emb_rank)."""
    frames = []
    s1_c, pool_c = s1["country_norm"].to_numpy(), pool["country_norm"].to_numpy()
    for country in pd.unique(s1_c):
        si = np.flatnonzero(s1_c == country)
        pi = np.flatnonzero(pool_c == country)
        if len(pi) == 0:
            pi = np.arange(len(pool))
        idx, sim = knn(s1_emb[si], pool_emb[pi], k)
        kk = idx.shape[1]
        frames.append(pd.DataFrame({
            "s1_idx": np.repeat(si, kk), "pool_idx": pi[idx.ravel()], "emb_score": sim.ravel(),
            "emb_rank": np.tile(np.arange(1, kk + 1, dtype=np.int16), len(si))}))
    return pd.concat(frames, ignore_index=True)
