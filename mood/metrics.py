"""Regression metrics reported separately for valence and arousal."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mood.data import TARGET_COLUMNS


def regression_metrics(targets, predictions) -> dict:
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    if targets.shape != predictions.shape or targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("targets and predictions must both have shape (n_samples, 2)")
    result = {}
    for idx, target in enumerate(TARGET_COLUMNS):
        result[f"{target}_mae"] = float(mean_absolute_error(targets[:, idx], predictions[:, idx]))
        result[f"{target}_rmse"] = float(mean_squared_error(targets[:, idx], predictions[:, idx]) ** 0.5)
        result[f"{target}_r2"] = float(r2_score(targets[:, idx], predictions[:, idx]))
    return result
