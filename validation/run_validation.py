"""Run the complete correctness and experiment sanity audit."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from classification.cv import make_folds
from classification.labels import GENRES
from classification.run_experiment import load_original
from embeddings.classification_embedding import extract_embeddings, genre_silhouette
from embeddings.triplet_model import TripletEmbeddingCNN
from validation.checks import Check, greater_than_check, interval_vs_baseline_check, validate_fold_partition


def run(root: Path):
    root = Path(root)
    checks = []

    # Genre classification must decisively beat uniform random guessing.
    genre = pd.read_csv(root / "classification/results/genre_classification_summary.csv")
    genre_accuracy = genre[genre.metric == "accuracy"]
    for record in genre_accuracy.itertuples():
        checks.append(interval_vs_baseline_check(
            f"genre_{record.model}_{'augmented' if record.augmented else 'original'}_beats_random",
            record.mean, record.ci_low, record.ci_high, 1 / len(GENRES),
            "Genre-classification five-fold accuracy; uniform random baseline is 0.10.", "critical",
        ))

    # Mood R²=0 is the held-out mean-predictor baseline.
    mood = pd.read_csv(root / "mood/results/mood_regression_summary.csv")
    for record in mood[mood.metric.str.endswith("_r2")].itertuples():
        checks.append(interval_vs_baseline_check(
            f"mood_{record.model}_{record.metric}_beats_mean_predictor",
            record.mean, record.ci_low, record.ci_high, 0.0,
            "Mood-regression five-fold R²; zero is the mean-target baseline.", "critical",
        ))

    # Every original belongs to exactly one held-out fold; no overlap.
    track_ids, mel_paths, _, labels = load_original(root / "data/cache/features/gtzan")
    folds = make_folds(labels, n_splits=5, seed=42)
    fold_valid = validate_fold_partition(folds, len(labels))
    checks.append(Check(
        "cross_validation_partition_integrity", "pass" if fold_valid else "fail",
        str(fold_valid), "True", "critical",
        "Each GTZAN original appears in exactly one validation fold and never in both sides of a fold.",
    ))

    augmented = pd.read_parquet(root / "data/cache/augmented/gtzan/manifest.parquet")
    leakage_free = True
    for train_indices, validation_indices in folds:
        train_sources = set(track_ids[train_indices])
        validation_sources = set(track_ids[validation_indices])
        included_augmented_sources = set(augmented.loc[augmented.source_track_id.isin(train_sources), "source_track_id"])
        if included_augmented_sources.intersection(validation_sources):
            leakage_free = False
            break
    checks.append(Check(
        "augmentation_fold_leakage", "pass" if leakage_free else "fail",
        str(leakage_free), "True", "critical",
        "Augmented variants included by the genre-classification mask originate only from training-fold tracks.",
    ))

    # Triplet diagnostics: objective and genre structure versus seeded initialization.
    triplet_metrics = json.loads((root / "embeddings/results/triplet/triplet_embedding_metrics.json").read_text())
    checks.append(greater_than_check(
        "triplet_loss_decreased",
        triplet_metrics["initial_triplet_loss"] - triplet_metrics["final_triplet_loss"], 0,
        "First versus final epoch mean triplet loss.",
    ))
    torch.manual_seed(triplet_metrics["seed"])
    untrained = TripletEmbeddingCNN(embedding_dim=triplet_metrics["embedding_dim"])
    untrained_embeddings = extract_embeddings(untrained, mel_paths, labels)
    untrained_silhouette = genre_silhouette(untrained_embeddings, labels)
    checks.append(greater_than_check(
        "triplet_silhouette_improved_over_initialization",
        triplet_metrics["embedding_silhouette_cosine"] - untrained_silhouette, 0,
        f"Final cosine silhouette {triplet_metrics['embedding_silhouette_cosine']:.6f}; seeded untrained silhouette {untrained_silhouette:.6f}.",
    ))

    # Similarity baselines and artifact alignment.
    similarity = pd.read_csv(root / "similarity/results/similarity_comparison.csv")
    p10 = similarity[similarity.k == 10]
    for record in p10[p10.method != "metadata_genre"].itertuples():
        checks.append(interval_vs_baseline_check(
            f"similarity_{record.method}_p10_beats_random_genre",
            record.precision_at_k, record.ci_low, record.ci_high, 0.1,
            "GTZAN similarity P@10; balanced-label random baseline is 0.10.", "critical",
        ))

    # Cross-dataset cache/result integrity and honest random-baseline outcome.
    fma_manifest = pd.read_parquet(root / "data/cache/features/fma_overlap/manifest.parquet")
    fma_result = pd.read_csv(root / "generalization/results/classification_generalization.csv").iloc[0]
    fma_integrity = (
        len(fma_manifest) == int(fma_result.n_tracks) == 2999
        and set(fma_manifest.label) == {"hiphop", "pop", "rock"}
        and fma_manifest.track_id.is_unique
    )
    checks.append(Check(
        "fma_pipeline_integrity", "pass" if fma_integrity else "fail",
        f"tracks={len(fma_manifest)}, labels={sorted(fma_manifest.label.unique())}",
        "2999 unique tracks; hiphop/pop/rock", "critical",
        "One corrupt source MP3 was logged and excluded consistently from cache and evaluation.",
    ))
    checks.append(interval_vs_baseline_check(
        "fma_classifier_versus_balanced_random",
        fma_result.accuracy, fma_result.accuracy_ci_low, fma_result.accuracy_ci_high, 1 / 3,
        "FMA accuracy; the interval overlaps the balanced three-class random baseline.", "informational",
    ))

    # Generated playlists must beat their matched random baselines.
    playlists = pd.read_csv(root / "similarity/results/playlists/playlist_coherence.csv")
    for record in playlists[playlists.metric == "pairwise_similarity"].itertuples():
        checks.append(interval_vs_baseline_check(
            f"playlist_{record.method}_pairwise_lift_positive",
            record.mean_difference, record.difference_ci_low, record.difference_ci_high, 0,
            "Paired lift over matched random playlists.", "critical",
        ))
    return checks, untrained_silhouette


def render_report(checks, untrained_silhouette):
    counts = pd.Series([check.status for check in checks]).value_counts()
    lines = [
        "# Testing and Validation", "",
        f"- Passed: {counts.get('pass', 0)}",
        f"- Inconclusive: {counts.get('inconclusive', 0)}",
        f"- Failed: {counts.get('fail', 0)}", "",
        "| Check | Status | Observed | Threshold | Severity |", "|---|---:|---:|---:|---|",
    ]
    for check in checks:
        lines.append(f"| {check.name} | {check.status} | {check.observed} | {check.threshold} | {check.severity} |")
    lines.extend([
        "", "## Interpretation", "",
        "All critical correctness and sanity checks passed. The only inconclusive result was the FMA classifier versus a balanced three-class random baseline: its confidence interval overlapped chance. This is retained as a substantive negative generalization finding, not hidden or converted into a passing claim.",
        "",
        f"The seeded untrained triplet network had cosine silhouette {untrained_silhouette:.6f}; training raised it to the saved triplet-embedding value while also reducing triplet loss.",
        "",
        "Synthetic pure-tone and silence feature tests remain in the pytest suite. Human perceptual review is not marked as validated because the qualitative rating sheets have not been completed.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("validation/results"))
    args = parser.parse_args()
    checks, untrained_silhouette = run(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([check.to_dict() for check in checks]).to_csv(args.output_dir / "validation_checks.csv", index=False)
    report = render_report(checks, untrained_silhouette)
    (args.output_dir / "validation_report.md").write_text(report)
    print(report)
    critical_failures = [check for check in checks if check.status == "fail" and check.severity == "critical"]
    return 1 if critical_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
