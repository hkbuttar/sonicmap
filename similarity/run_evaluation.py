"""Run four-way audio similarity search and precision@k evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from classification.run_experiment import load_original
from similarity.evaluation import (
    bootstrap_mean_ci, metadata_neighbors, nearest_neighbors, precision_at_k,
)


def _load_embedding(embedding_path, projection_path, expected_track_ids):
    vectors = np.load(embedding_path)
    metadata = pd.read_csv(projection_path)
    if metadata["track_id"].tolist() != list(expected_track_ids):
        raise ValueError(f"Track ordering in {embedding_path} does not match the GTZAN feature cache")
    if len(vectors) != len(metadata):
        raise ValueError(f"Embedding and metadata row counts differ for {embedding_path}")
    return vectors


def _neighbor_rows(name, metric, indices, distances, track_ids, labels):
    rows = []
    for query_idx in range(len(track_ids)):
        for rank, neighbor_idx in enumerate(indices[query_idx], start=1):
            rows.append({
                "method": name, "distance_metric": metric,
                "query_track_id": track_ids[query_idx], "query_label": labels[query_idx],
                "rank": rank, "neighbor_track_id": track_ids[neighbor_idx],
                "neighbor_label": labels[neighbor_idx],
                "distance": float(distances[query_idx, rank - 1]),
                "same_genre": bool(labels[query_idx] == labels[neighbor_idx]),
            })
    return rows


def _qualitative_sheet(neighbors, labels, seed, n_queries_per_genre=1):
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(np.unique(labels)):
        candidates = np.flatnonzero(labels == label)
        selected.extend(rng.choice(candidates, size=min(n_queries_per_genre, len(candidates)), replace=False))
    sheet = neighbors[(neighbors["query_index"].isin(selected)) & (neighbors["rank"] <= 5)].copy()
    sheet["human_similarity_1_to_5"] = ""
    sheet["human_notes"] = ""
    return sheet.drop(columns="query_index")


def run(features_dir, classification_dir, triplet_dir, output_dir, ks, max_k, n_bootstrap, seed):
    track_ids, _, engineered, label_indices = load_original(features_dir)
    labels = np.asarray([track_id.split("/", 1)[0] for track_id in track_ids])
    classification = _load_embedding(
        Path(classification_dir) / "classification_embeddings.npy",
        Path(classification_dir) / "classification_projection.csv", track_ids,
    )
    triplet = _load_embedding(
        Path(triplet_dir) / "triplet_embeddings.npy",
        Path(triplet_dir) / "triplet_projection.csv", track_ids,
    )
    engineered = StandardScaler().fit_transform(engineered)

    configurations = []
    for name, vectors in (
        ("classification_embedding", classification),
        ("triplet_embedding", triplet),
        ("engineered_features", engineered),
    ):
        for metric in ("cosine", "euclidean"):
            indices, distances = nearest_neighbors(vectors, metric=metric, max_k=max_k)
            configurations.append((name, metric, indices, distances))
    metadata_idx, metadata_dist = metadata_neighbors(label_indices, max_k=max_k)
    configurations.append(("metadata_genre", "label_exact_match", metadata_idx, metadata_dist))

    metric_rows, neighbor_rows = [], []
    for config_idx, (name, metric, indices, distances) in enumerate(configurations):
        for k in ks:
            per_query = precision_at_k(indices, label_indices, k)
            mean, low, high = bootstrap_mean_ci(per_query, n_bootstrap=n_bootstrap, seed=seed + config_idx * 100 + k)
            metric_rows.append({
                "method": name, "distance_metric": metric, "k": k,
                "precision_at_k": mean, "ci_low": low, "ci_high": high,
                "n_queries": len(track_ids), "n_bootstrap": n_bootstrap,
            })
        rows = _neighbor_rows(name, metric, indices, distances, track_ids, labels)
        for query_idx, row in enumerate(rows):
            row["query_index"] = query_idx // max_k
        neighbor_rows.extend(rows)

    metrics = pd.DataFrame(metric_rows)
    neighbors = pd.DataFrame(neighbor_rows)
    primary_metrics = {
        "classification_embedding": "cosine", "triplet_embedding": "cosine",
        "engineered_features": "cosine", "metadata_genre": "label_exact_match",
    }
    comparison = pd.concat([
        metrics[(metrics["method"] == method) & (metrics["distance_metric"] == distance)]
        for method, distance in primary_metrics.items()
    ], ignore_index=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "similarity_comparison.csv", index=False)
    metrics.to_csv(output_dir / "distance_comparison.csv", index=False)
    neighbors.drop(columns="query_index").to_parquet(output_dir / "neighbors.parquet", index=False)
    qualitative = _qualitative_sheet(neighbors, labels, seed)
    qualitative.to_csv(output_dir / "qualitative_review.csv", index=False)
    protocol = {
        "automated_relevance_signal": "same GTZAN genre label",
        "warning": "Genre agreement is a weak proxy for perceptual similarity; metadata_genre is an oracle-like upper bound, not audio retrieval.",
        "evaluation_scope": "In-sample retrieval on the original GTZAN tracks used for full-data embedding training; FMA provides the out-of-distribution test.",
        "primary_distance": "cosine for audio representations; exact label match for metadata",
        "ks": list(ks), "max_neighbors_saved": max_k,
        "qualitative_queries": int(qualitative["query_track_id"].nunique()),
        "seed": seed,
    }
    (output_dir / "evaluation_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("data/cache/features/gtzan"))
    parser.add_argument("--classification-dir", type=Path, default=Path("embeddings/results/classification"))
    parser.add_argument("--triplet-dir", type=Path, default=Path("embeddings/results/triplet"))
    parser.add_argument("--output-dir", type=Path, default=Path("similarity/results"))
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if any(k < 1 for k in args.ks):
        parser.error("all k values must be positive")
    comparison = run(
        args.features_dir, args.classification_dir, args.triplet_dir, args.output_dir,
        tuple(sorted(set(args.ks))), max(args.ks), args.bootstrap_samples, args.seed,
    )
    print("\n=== Four-way similarity comparison ===")
    print(comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
