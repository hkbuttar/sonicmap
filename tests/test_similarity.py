import numpy as np

from similarity.evaluation import (
    bootstrap_mean_ci, metadata_neighbors, nearest_neighbors, precision_at_k,
)


def test_nearest_neighbors_excludes_query_itself():
    vectors = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
    indices, _ = nearest_neighbors(vectors, max_k=1)
    assert indices[:, 0].tolist() == [1, 0, 1]


def test_precision_at_k_uses_query_genre():
    neighbors = np.array([[1, 2], [0, 2], [1, 0]])
    scores = precision_at_k(neighbors, np.array([0, 0, 1]), k=2)
    np.testing.assert_allclose(scores, [0.5, 0.5, 0.0])


def test_metadata_baseline_ranks_same_label_first():
    labels = np.array([0, 0, 1, 1])
    indices, _ = metadata_neighbors(labels, max_k=1)
    assert precision_at_k(indices, labels, 1).tolist() == [1.0] * 4


def test_bootstrap_constant_values_have_zero_width_interval():
    mean, low, high = bootstrap_mean_ci(np.ones(20), n_bootstrap=100)
    assert (mean, low, high) == (1.0, 1.0, 1.0)
