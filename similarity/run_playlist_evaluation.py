"""Step 10: generate embedding-traversal playlists and evaluate coherence."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from classification.run_experiment import load_original
from similarity.evaluation import bootstrap_mean_ci
from similarity.playlist import cosine_similarity_matrix, generate_progressive_playlist, playlist_metrics
from similarity.run_evaluation import _load_embedding

METRIC_NAMES = (
    "pairwise_similarity", "adjacent_similarity", "seed_genre_fraction",
    "monotonic_drift_fraction", "final_seed_distance",
)


def stratified_seeds(labels, seeds_per_genre, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        selected.extend(rng.choice(candidates, size=min(seeds_per_genre, len(candidates)), replace=False))
    return np.asarray(selected, dtype=int)


def _random_metrics(seed_index, length, similarity, labels, samples, rng):
    candidates = np.delete(np.arange(len(labels)), seed_index)
    records = []
    for _ in range(samples):
        playlist = [seed_index, *rng.choice(candidates, size=length - 1, replace=False)]
        records.append(playlist_metrics(playlist, similarity, labels, seed_index))
    return {name: float(np.mean([record[name] for record in records])) for name in METRIC_NAMES}


def _playlist_rows(method, seed_index, playlist, similarity, track_ids, labels):
    rows = []
    for position, track_index in enumerate(playlist, start=1):
        rows.append({
            "method": method, "seed_track_id": track_ids[seed_index],
            "seed_label": labels[seed_index], "position": position,
            "track_id": track_ids[track_index], "track_label": labels[track_index],
            "similarity_to_seed": float(similarity[seed_index, track_index]),
            "similarity_to_previous": 1.0 if position == 1 else float(similarity[playlist[position - 2], track_index]),
        })
    return rows


def run(features_dir, classification_dir, triplet_dir, output_dir, length, seeds_per_genre,
        random_samples, drift_quantile, drift_weight, seed, requested_seed_tracks=None):
    track_ids, _, _, labels = load_original(features_dir)
    genre_labels = np.asarray([track_id.split("/", 1)[0] for track_id in track_ids])
    representations = {
        "classification_embedding": _load_embedding(
            Path(classification_dir) / "classification_embeddings.npy",
            Path(classification_dir) / "classification_projection.csv", track_ids,
        ),
        "triplet_embedding": _load_embedding(
            Path(triplet_dir) / "triplet_embeddings.npy",
            Path(triplet_dir) / "triplet_projection.csv", track_ids,
        ),
    }
    evaluation_seeds = stratified_seeds(labels, seeds_per_genre, seed)
    if requested_seed_tracks:
        lookup = {track_id: idx for idx, track_id in enumerate(track_ids)}
        missing = sorted(set(requested_seed_tracks).difference(lookup))
        if missing:
            raise ValueError(f"Unknown seed track IDs: {missing}")
        demo_seeds = np.asarray([lookup[value] for value in requested_seed_tracks])
    else:
        demo_seeds = stratified_seeds(labels, 1, seed)

    rng = np.random.default_rng(seed)
    evaluation_rows, demo_rows = [], []
    for method, embeddings in representations.items():
        similarity = cosine_similarity_matrix(embeddings)
        for seed_index in evaluation_seeds:
            playlist = generate_progressive_playlist(
                similarity, seed_index, length, drift_quantile, drift_weight,
            )
            generated = playlist_metrics(playlist, similarity, labels, seed_index)
            random = _random_metrics(seed_index, length, similarity, labels, random_samples, rng)
            evaluation_rows.append({
                "method": method, "seed_track_id": track_ids[seed_index],
                **{f"playlist_{name}": generated[name] for name in METRIC_NAMES},
                **{f"random_{name}": random[name] for name in METRIC_NAMES},
            })
        for seed_index in demo_seeds:
            playlist = generate_progressive_playlist(
                similarity, seed_index, length, drift_quantile, drift_weight,
            )
            demo_rows.extend(_playlist_rows(method, seed_index, playlist, similarity, track_ids, genre_labels))

    evaluation = pd.DataFrame(evaluation_rows)
    summaries = []
    for method, group in evaluation.groupby("method"):
        for metric in METRIC_NAMES:
            generated = group[f"playlist_{metric}"].to_numpy()
            random_values = group[f"random_{metric}"].to_numpy()
            differences = generated - random_values
            lift, low, high = bootstrap_mean_ci(differences, seed=seed)
            summaries.append({
                "method": method, "metric": metric,
                "playlist_mean": float(generated.mean()), "random_mean": float(random_values.mean()),
                "mean_difference": lift, "difference_ci_low": low, "difference_ci_high": high,
                "n_seed_tracks": len(group), "random_playlists_per_seed": random_samples,
            })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "step10_playlist_coherence.csv", index=False)
    evaluation.to_csv(output_dir / "step10_playlist_per_seed.csv", index=False)
    demos = pd.DataFrame(demo_rows)
    demos.to_csv(output_dir / "step10_generated_playlists.csv", index=False)
    review = demos.copy()
    review["human_transition_rating_1_to_5"] = ""
    review["human_notes"] = ""
    review.to_csv(output_dir / "step10_qualitative_review.csv", index=False)
    protocol = {
        "strategy": "greedy local traversal with a linearly increasing target distance from the seed",
        "playlist_length": length, "drift_quantile": drift_quantile,
        "drift_weight": drift_weight, "evaluation_seed_tracks": int(len(evaluation_seeds)),
        "random_playlists_per_seed": random_samples,
        "coherence_primary_metric": "average pairwise cosine similarity",
        "comparison_note": "Compare each method's lift over its own random baseline; absolute cosine levels are not directly comparable across separately trained embedding spaces.",
        "qualitative_note": "Human review remains necessary; genre and embedding similarity do not fully capture playlist flow.",
        "seed": seed,
    }
    (output_dir / "step10_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return summary, demos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("data/cache/features/gtzan"))
    parser.add_argument("--classification-dir", type=Path, default=Path("embeddings/results/classification"))
    parser.add_argument("--triplet-dir", type=Path, default=Path("embeddings/results/triplet"))
    parser.add_argument("--output-dir", type=Path, default=Path("similarity/results/playlists"))
    parser.add_argument("--length", type=int, default=10)
    parser.add_argument("--seeds-per-genre", type=int, default=10)
    parser.add_argument("--random-samples", type=int, default=50)
    parser.add_argument("--drift-quantile", type=float, default=0.25)
    parser.add_argument("--drift-weight", type=float, default=1.0)
    parser.add_argument("--seed-track", action="append", dest="seed_tracks")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary, demos = run(
        args.features_dir, args.classification_dir, args.triplet_dir, args.output_dir,
        args.length, args.seeds_per_genre, args.random_samples, args.drift_quantile,
        args.drift_weight, args.seed, args.seed_tracks,
    )
    print("\n=== Playlist coherence versus random ===")
    print(summary.to_string(index=False))
    print(f"\nGenerated {demos['seed_track_id'].nunique()} seed playlists per embedding in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
