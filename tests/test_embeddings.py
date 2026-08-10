import numpy as np
import pytest

from embeddings.classification_embedding import genre_silhouette, l2_normalize, project_2d
from embeddings.triplet_dataset import TripletMelDataset
from embeddings.triplet_model import TripletEmbeddingCNN


def test_l2_normalize_handles_zero_vectors():
    result = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(result[0], [0.6, 0.8])
    np.testing.assert_allclose(result[1], [0.0, 0.0])


def test_genre_silhouette_detects_separated_clusters():
    embeddings = np.array([[1, 0.01], [1, -0.01], [0.01, 1], [-0.01, 1]], dtype=np.float32)
    assert genre_silhouette(embeddings, np.array([0, 0, 1, 1])) > 0.99


def test_projection_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown projection method"):
        project_2d(np.eye(4), method="pca")


def test_triplet_sampler_respects_labels_and_distinct_sources(tmp_path):
    paths = [tmp_path / f"{idx}.npy" for idx in range(6)]
    dataset = TripletMelDataset(paths, [0, 0, 0, 1, 1, 1], ["a", "a", "b", "c", "d", "e"])
    anchor, positive, negative = dataset.sample_indices(0)
    assert dataset.labels[anchor] == dataset.labels[positive]
    assert dataset.source_ids[anchor] != dataset.source_ids[positive]
    assert dataset.labels[anchor] != dataset.labels[negative]


def test_triplet_model_outputs_unit_norm_embeddings():
    import torch

    model = TripletEmbeddingCNN(embedding_dim=8)
    embeddings = model.embed(torch.randn(2, 1, 32, 64))
    torch.testing.assert_close(torch.linalg.vector_norm(embeddings, dim=1), torch.ones(2))
