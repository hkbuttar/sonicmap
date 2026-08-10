"""Extraction, projection, and evaluation of genre-CNN embeddings."""

import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

from classification.dataset import MelDataset


def extract_embeddings(model, mel_paths, labels, batch_size: int = 64) -> np.ndarray:
    """Extract the CNN's penultimate activations in input order."""
    dataset = MelDataset(mel_paths, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batches = []
    model.eval()
    with torch.no_grad():
        for audio, _ in loader:
            batches.append(model.embed(audio).cpu().numpy())
    return np.concatenate(batches).astype(np.float32)


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, np.finfo(np.float32).eps)


def genre_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Cosine silhouette score, matching downstream similarity search."""
    if len(np.unique(labels)) < 2:
        raise ValueError("silhouette score requires at least two genres")
    return float(silhouette_score(l2_normalize(embeddings), labels, metric="cosine"))


def project_2d(embeddings: np.ndarray, method: str = "umap", seed: int = 42) -> np.ndarray:
    normalized = l2_normalize(embeddings)
    if method == "umap":
        import umap

        projector = umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1,
            metric="cosine", random_state=seed, n_jobs=1,
        )
    elif method == "tsne":
        projector = TSNE(
            n_components=2, perplexity=min(30, max(2, (len(normalized) - 1) // 3)),
            init="pca", learning_rate="auto", random_state=seed,
        )
    else:
        raise ValueError(f"Unknown projection method: {method}")
    return projector.fit_transform(normalized).astype(np.float32)
