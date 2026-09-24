"""Decision layer: pair probabilities -> final match set per S1 entity."""


def one_to_one(pair_probs):
    """Keep each S2/S3 record only under the S1 with its highest probability."""
    raise NotImplementedError


def s2_s3_boost(pair_probs, cfg):
    """Boost S3 candidates that look like an already-confident S2 match (and vice versa)."""
    raise NotImplementedError


def choose_set(cands, p_none_weight=1.0):
    """cands: [(entity_id, prob)] sorted desc. Return the prefix (possibly empty)
    that maximises expected F0.5."""
    raise NotImplementedError


def decide(pair_probs, cfg):
    """Probabilities -> {s1_id: [matched ids]}."""
    raise NotImplementedError
