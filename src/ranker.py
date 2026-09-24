"""LightGBM pair scorer, stacked on out-of-fold cross-encoder probabilities."""


def train_lgbm(features, labels, groups, cfg):
    """Train with GroupKFold by S1 id; return models and OOF predictions."""
    raise NotImplementedError


def predict_lgbm(features, ce_prob):
    """Features + CE prob -> match probability per pair."""
    raise NotImplementedError


def calibrate(probs, labels):
    """Fit a calibrator so probabilities are usable by the decision layer."""
    raise NotImplementedError
