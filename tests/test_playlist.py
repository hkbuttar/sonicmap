import numpy as np

from similarity.playlist import cosine_similarity_matrix, generate_progressive_playlist, playlist_metrics


def test_progressive_playlist_is_unique_and_starts_with_seed():
    vectors = np.array([[1, 0], [.9, .1], [.7, .3], [0, 1]], dtype=np.float32)
    similarity = cosine_similarity_matrix(vectors)
    playlist = generate_progressive_playlist(similarity, seed_index=1, length=4)
    assert playlist[0] == 1
    assert len(playlist) == len(set(playlist)) == 4


def test_pairwise_coherence_is_higher_for_identical_vectors():
    vectors = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32)
    similarity = cosine_similarity_matrix(vectors)
    labels = np.array([0, 0, 0, 1])
    coherent = playlist_metrics([0, 1, 2], similarity, labels)
    mixed = playlist_metrics([0, 1, 3], similarity, labels)
    assert coherent["pairwise_similarity"] > mixed["pairwise_similarity"]
    assert coherent["seed_genre_fraction"] == 1


def test_cosine_similarity_normalizes_inputs():
    similarity = cosine_similarity_matrix(np.array([[2, 0], [4, 0]], dtype=np.float32))
    np.testing.assert_allclose(similarity, np.ones((2, 2)))
