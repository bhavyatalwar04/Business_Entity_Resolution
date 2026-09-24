"""Pair features for LightGBM."""


def build_pair_features(pairs, s1, s23):
    """Candidate pairs -> numeric feature matrix (name/address similarity,
    postcode/number agreement, embedding cosine, rank and score-gap context)."""
    raise NotImplementedError
