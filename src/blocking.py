"""Candidate generation.

Output: DataFrame(s1_id, cand_id, block_source, block_score, block_rank).
"""


def tfidf_block(s1, s23, cfg):
    """Pass A: char 3-5-gram TF-IDF on cleaned names -> top-k neighbours."""
    raise NotImplementedError


def embed_block(s1, s23, cfg):
    """Pass B: multilingual embeddings of 'name | address' + FAISS -> top-k."""
    raise NotImplementedError


def token_block(s1, s23, cfg):
    """Pass C: shared rare name tokens, and same postcode + some name overlap."""
    raise NotImplementedError


def union_candidates(s1, s23, cfg):
    """Union all passes, dedupe, cap at cfg max_candidates per S1."""
    raise NotImplementedError
