from pathlib import Path

import numpy as np

from generalization.run_evaluation import adjusted_precision, per_genre_accuracy
from generalization.taxonomy import FMA_TO_GTZAN, fma_audio_path


def test_taxonomy_only_contains_exact_overlap():
    assert FMA_TO_GTZAN == {"Hip-Hop": "hiphop", "Pop": "pop", "Rock": "rock"}


def test_fma_audio_path_uses_official_layout():
    assert fma_audio_path(Path("audio"), 2) == Path("audio/000/000002.mp3")
    assert fma_audio_path(Path("audio"), 12345) == Path("audio/012/012345.mp3")


def test_adjusted_precision_removes_random_baseline():
    assert adjusted_precision(1 / 3, 1 / 3) == 0
    assert adjusted_precision(1, 1 / 3) == 1
    np.testing.assert_allclose(adjusted_precision(2 / 3, 1 / 3), 0.5)


def test_per_genre_accuracy_keeps_group_label():
    result = per_genre_accuracy(np.array([4, 4, 7]), np.array([4, 7, 7]))
    assert result[["label", "n", "accuracy", "genre"]].to_dict("records") == [
        {"label": 4, "n": 2, "accuracy": 0.5, "genre": "hiphop"},
        {"label": 7, "n": 1, "accuracy": 1.0, "genre": "pop"},
    ]
