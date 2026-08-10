"""Exact neighbor search and precision-at-k evaluation utilities."""

import numpy as np
from sklearn.metrics import pairwise_distances


def nearest_neighbors(vectors, metric="cosine", max_k=20):
    """Return exact neighbor indices/distances, always excluding self."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if len(vectors) <= max_k:
        raise ValueError("max_k must be smaller than the number of tracks")
    distances = pairwise_distances(vectors, metric=metric)
    np.fill_diagonal(distances, np.inf)
    # Stable sorting makes tied metadata distances reproducible.
    indices = np.argsort(distances, axis=1, kind="stable")[:, :max_k]
    return indices, np.take_along_axis(distances, indices, axis=1)


def metadata_neighbors(labels, max_k=20):
    """Oracle-like genre-label ranking used only as a metadata baseline."""
    labels = np.asarray(labels)
    distances = (labels[:, None] != labels[None, :]).astype(np.float32)
    np.fill_diagonal(distances, np.inf)
    indices = np.argsort(distances, axis=1, kind="stable")[:, :max_k]
    return indices, np.take_along_axis(distances, indices, axis=1)


def precision_at_k(neighbor_indices, labels, k):
    labels = np.asarray(labels)
    neighbors = np.asarray(neighbor_indices)[:, :k]
    return (labels[neighbors] == labels[:, None]).mean(axis=1)


def bootstrap_mean_ci(values, confidence=0.95, n_bootstrap=2000, seed=42):
    """Percentile bootstrap CI over query-level precision values."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap)
    for start in range(0, n_bootstrap, 250):
        count = min(250, n_bootstrap - start)
        samples = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[samples].mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(values.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))
