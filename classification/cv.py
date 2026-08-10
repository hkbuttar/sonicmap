"""Cross-validation utilities: stratified folds over *original* tracks
only, plus confidence intervals for fold-level metric aggregation.

Augmented clips are never split into folds directly — a fold is defined
purely over the 999 original GTZAN tracks, and a training fold is
optionally expanded with the augmented siblings of whichever original
tracks landed in that fold's training half. This guarantees no track's
augmented variant can appear in a different fold than the track itself,
which would otherwise leak near-duplicate information across the
train/validation boundary and inflate held-out accuracy.
"""

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold


def make_folds(labels: np.ndarray, n_splits: int = 5, seed: int = 42) -> list:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(labels)), labels))


def mean_ci(scores, confidence: float = 0.95) -> tuple:
    """Mean and a t-distribution confidence interval across fold scores.
    Uses the t-distribution (not normal) because k-fold CV yields very few
    samples (typically 5), where the normal approximation understates
    uncertainty."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    mean = float(scores.mean())
    if n < 2:
        return mean, mean, mean
    se = scores.std(ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h
