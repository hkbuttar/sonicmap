"""Controlled embedding-space playlist traversal and coherence metrics."""

import numpy as np


def cosine_similarity_matrix(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, np.finfo(np.float32).eps)
    return normalized @ normalized.T


def generate_progressive_playlist(similarity, seed_index, length=10, drift_quantile=0.25, drift_weight=1.0):
    """Walk locally while moving gradually farther from the original seed."""
    similarity = np.asarray(similarity)
    n_tracks = len(similarity)
    if similarity.shape != (n_tracks, n_tracks):
        raise ValueError("similarity must be a square matrix")
    if not 1 <= length <= n_tracks:
        raise ValueError("playlist length must be between 1 and the number of tracks")
    if not 0 < drift_quantile <= 1:
        raise ValueError("drift_quantile must be in (0, 1]")

    seed_distances = 1 - similarity[seed_index]
    non_seed = np.arange(n_tracks) != seed_index
    maximum_drift = float(np.quantile(seed_distances[non_seed], drift_quantile))
    playlist = [int(seed_index)]
    used = np.zeros(n_tracks, dtype=bool)
    used[seed_index] = True
    for position in range(1, length):
        candidates = np.flatnonzero(~used)
        target_drift = maximum_drift * position / max(1, length - 1)
        continuity_cost = 1 - similarity[playlist[-1], candidates]
        drift_cost = np.abs(seed_distances[candidates] - target_drift)
        score = continuity_cost + drift_weight * drift_cost
        chosen = int(candidates[np.argmin(score)])
        playlist.append(chosen)
        used[chosen] = True
    return playlist


def playlist_metrics(indices, similarity, labels, seed_index=None):
    indices = np.asarray(indices, dtype=int)
    submatrix = similarity[np.ix_(indices, indices)]
    upper = submatrix[np.triu_indices(len(indices), k=1)]
    adjacent = similarity[indices[:-1], indices[1:]] if len(indices) > 1 else np.array([1.0])
    seed_index = int(indices[0] if seed_index is None else seed_index)
    seed_distances = 1 - similarity[seed_index, indices]
    drift_fraction = float(np.mean(np.diff(seed_distances) >= 0)) if len(indices) > 1 else 1.0
    return {
        "pairwise_similarity": float(upper.mean()) if len(upper) else 1.0,
        "adjacent_similarity": float(adjacent.mean()),
        "seed_genre_fraction": float(np.mean(np.asarray(labels)[indices] == np.asarray(labels)[seed_index])),
        "monotonic_drift_fraction": drift_fraction,
        "final_seed_distance": float(seed_distances[-1]),
    }
