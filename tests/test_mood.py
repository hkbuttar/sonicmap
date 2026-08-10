import numpy as np
import pandas as pd

from mood.data import load_deam_labels
from mood.metrics import regression_metrics
from mood.run_experiment import summarize


def test_load_deam_labels_aligns_schema(tmp_path):
    pd.DataFrame({"song_id": [2], "valence_mean": [3.1], "arousal_mean": [3.0]}).to_csv(
        tmp_path / "static_annotations_averaged_songs_1_2000.csv", index=False
    )
    labels = load_deam_labels(tmp_path)
    assert labels.to_dict("records") == [{"track_id": "2", "valence": 3.1, "arousal": 3.0}]


def test_regression_metrics_are_per_target():
    targets = np.array([[1.0, 2.0], [3.0, 4.0]])
    metrics = regression_metrics(targets, targets)
    assert metrics["valence_mae"] == 0
    assert metrics["arousal_rmse"] == 0
    assert metrics["valence_r2"] == 1


def test_summary_has_confidence_interval_for_every_metric():
    row = {"fold": 0, "model": "cnn", **{k: 0.5 for k in (
        "valence_mae", "valence_rmse", "valence_r2", "arousal_mae", "arousal_rmse", "arousal_r2")}}
    summary = summarize(pd.DataFrame([row]))
    assert len(summary) == 6
    assert (summary["ci_low"] == summary["ci_high"]).all()
