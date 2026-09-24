"""Local scoring: exact per-entity F0.5, macro-averaged over all S1 (singletons included)."""


def f05(pred, gold):
    """Official per-entity F0.5 score."""
    raise NotImplementedError


def macro_f05(pred, gold):
    """Mean f05 over every S1 in gold. pred/gold: {s1_id: set(ids)}."""
    raise NotImplementedError


def blocking_recall(candidates, gold):
    """Share of true matches that appear in the candidate set."""
    raise NotImplementedError


def loco_eval(cfg):
    """Leave-one-country-out: train on one country, score on the other."""
    raise NotImplementedError
