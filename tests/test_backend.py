from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def test_health_is_ok():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_embedding_endpoint_returns_all_gtzan_points():
    response = client.get("/api/embeddings/classification")
    assert response.status_code == 200
    assert len(response.json()["points"]) == 999


def test_summary_is_derived_from_compiled_results():
    response = client.get("/api/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["headline"]["genre_accuracy"] > body["headline"]["fma_accuracy"]
    assert len(body["comparisons"]) >= 20


def test_similarity_endpoint_excludes_seed():
    track_id = "blues/blues.00000"
    response = client.get(f"/api/similarity/{track_id}?k=5")
    assert response.status_code == 200
    neighbors = response.json()["neighbors"]
    assert len(neighbors) == 5
    assert track_id not in {row["neighbor_track_id"] for row in neighbors}


def test_playlist_endpoint_generates_unique_tracks():
    response = client.get("/api/playlists/blues/blues.00000?space=triplet&length=8")
    assert response.status_code == 200
    tracks = response.json()["tracks"]
    assert len(tracks) == 8
    assert len({row["track_id"] for row in tracks}) == 8


def test_unknown_embedding_space_is_404():
    assert client.get("/api/embeddings/unknown").status_code == 404
