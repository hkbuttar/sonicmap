"""Read-only API over SonicMap's reproducible experiment artifacts."""

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from similarity.playlist import cosine_similarity_matrix, generate_progressive_playlist

ROOT = Path(os.getenv("SONICMAP_ROOT", Path(__file__).resolve().parent.parent)).resolve()

app = FastAPI(
    title="SonicMap API",
    description="Genre, mood, embedding, similarity, generalization, and playlist experiment results.",
    version="1.0.0",
)
origins = [value.strip() for value in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if value.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=False,
    allow_methods=["GET"], allow_headers=["*"],
)


def _csv(relative_path):
    path = ROOT / relative_path
    if not path.exists():
        raise HTTPException(503, f"Required artifact is unavailable: {relative_path}")
    return pd.read_csv(path)


def _records(frame):
    return json.loads(frame.to_json(orient="records"))


@lru_cache(maxsize=1)
def _projections():
    result = {}
    for space, directory, prefix in (
        ("classification", "classification", "classification"),
        ("triplet", "triplet", "triplet"),
    ):
        metadata = pd.read_csv(ROOT / f"embeddings/results/{directory}/{prefix}_projection.csv")
        embeddings = np.load(ROOT / f"embeddings/results/{directory}/{prefix}_embeddings.npy")
        result[space] = (metadata, embeddings)
    return result


@lru_cache(maxsize=1)
def _neighbors():
    path = ROOT / "similarity/results/neighbors.parquet"
    if not path.exists():
        raise HTTPException(503, "Similarity neighbor artifact is unavailable")
    return pd.read_parquet(path)


@lru_cache(maxsize=2)
def _similarity_matrix(space):
    return cosine_similarity_matrix(_projections()[space][1])


@app.get("/")
def root():
    return {"name": "SonicMap API", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health")
def health():
    required = [
        "results/comparison.csv",
        "embeddings/results/classification/classification_projection.csv",
        "embeddings/results/triplet/triplet_projection.csv",
        "similarity/results/neighbors.parquet",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    return {"status": "ok" if not missing else "degraded", "missing_artifacts": missing}


@app.get("/api/summary")
def summary():
    comparison = _csv("results/comparison.csv")
    def value(section, name, metric):
        row = comparison[
            (comparison.section == section) & (comparison.comparison == name)
            & (comparison.metric == metric)
        ]
        return float(row.value.iloc[0])
    return {
        "headline": {
            "genre_accuracy": value("genre_classification", "cnn", "accuracy"),
            "mood_arousal_r2": value("mood_regression", "gbm", "arousal_r2"),
            "classification_precision_at_10": value("similarity", "classification_embedding", "precision_at_10"),
            "fma_accuracy": value("generalization", "genre_cnn", "fma_accuracy"),
            "classification_playlist_lift": value("playlist", "classification_embedding", "pairwise_similarity_lift_vs_random"),
        },
        "comparisons": _records(comparison),
        "disclosure": "Similarity uses genre agreement as a weak proxy; FMA is the unseen-distribution test.",
    }


@app.get("/api/classification")
def classification_results():
    return _records(_csv("classification/results/genre_classification_summary.csv"))


@app.get("/api/mood")
def mood_results():
    return _records(_csv("mood/results/mood_regression_summary.csv"))


@app.get("/api/embeddings/{space}")
def embedding_projection(space: str):
    if space not in _projections():
        raise HTTPException(404, "Embedding space must be 'classification' or 'triplet'")
    metadata, _ = _projections()[space]
    metrics_path = ROOT / f"embeddings/results/{space}/{'classification' if space == 'classification' else 'triplet'}_embedding_metrics.json"
    return {"space": space, "metrics": json.loads(metrics_path.read_text()), "points": _records(metadata)}


@app.get("/api/tracks")
def tracks(search: str | None = None, limit: int = Query(100, ge=1, le=1000)):
    metadata = _projections()["classification"][0][["track_id", "label"]]
    if search:
        metadata = metadata[metadata.track_id.str.contains(search, case=False, regex=False)]
    return _records(metadata.head(limit))


@app.get("/api/similarity/{track_id:path}")
def similarity_search(
    track_id: str, method: str = "classification_embedding",
    distance_metric: str | None = None, k: int = Query(10, ge=1, le=20),
):
    allowed = {
        "classification_embedding": "cosine", "triplet_embedding": "cosine",
        "engineered_features": "cosine", "metadata_genre": "label_exact_match",
    }
    if method not in allowed:
        raise HTTPException(400, f"Unknown method. Choose from {sorted(allowed)}")
    metric = distance_metric or allowed[method]
    matches = _neighbors()
    matches = matches[
        (matches.query_track_id == track_id) & (matches.method == method)
        & (matches.distance_metric == metric) & (matches["rank"] <= k)
    ]
    if matches.empty:
        known = track_id in set(_projections()["classification"][0].track_id)
        raise HTTPException(404 if not known else 400, "Track or method/distance combination was not found")
    return {"track_id": track_id, "method": method, "distance_metric": metric, "neighbors": _records(matches)}


@app.get("/api/generalization")
def generalization_results():
    return {
        "classification": _records(_csv("generalization/results/classification_generalization.csv")),
        "per_genre": _records(_csv("generalization/results/per_genre_accuracy.csv")),
        "similarity": _records(_csv("generalization/results/similarity_generalization.csv")),
    }


@app.get("/api/playlists/{track_id:path}")
def generate_playlist(
    track_id: str, space: str = "classification", length: int = Query(10, ge=2, le=30),
    drift_quantile: float = Query(0.25, gt=0, le=1),
):
    if space not in _projections():
        raise HTTPException(400, "Space must be 'classification' or 'triplet'")
    metadata, embeddings = _projections()[space]
    lookup = {value: idx for idx, value in enumerate(metadata.track_id)}
    if track_id not in lookup:
        raise HTTPException(404, "Unknown track ID")
    similarity = _similarity_matrix(space)
    seed_index = lookup[track_id]
    indices = generate_progressive_playlist(similarity, seed_index, length, drift_quantile)
    rows = []
    for position, index in enumerate(indices, start=1):
        rows.append({
            "position": position, "track_id": metadata.iloc[index].track_id,
            "label": metadata.iloc[index].label,
            "similarity_to_seed": float(similarity[seed_index, index]),
            "similarity_to_previous": 1.0 if position == 1 else float(similarity[indices[position - 2], index]),
        })
    return {"seed_track_id": track_id, "space": space, "drift_quantile": drift_quantile, "tracks": rows}


@app.get("/api/validation")
def validation_results():
    checks = _csv("validation/results/validation_checks.csv")
    return {"counts": checks.status.value_counts().to_dict(), "checks": _records(checks)}
