"""Compile completed experiments into an honest, reproducible report."""

import argparse
import json
from pathlib import Path

import pandas as pd

from classification.cv import mean_ci


def _row(section, comparison, metric, value, low=None, high=None, source="", caveat=""):
    return {
        "section": section, "comparison": comparison, "metric": metric,
        "value": float(value), "ci_low": None if low is None else float(low),
        "ci_high": None if high is None else float(high), "source": source,
        "caveat": caveat,
    }


def compile_results(root: Path):
    root = Path(root)
    genre_raw = pd.read_csv(root / "classification/results/genre_classification.csv")
    genre_results = pd.read_csv(root / "classification/results/genre_classification_summary.csv")
    mood_raw = pd.read_csv(root / "mood/results/mood_regression.csv")
    mood = pd.read_csv(root / "mood/results/mood_regression_summary.csv")
    similarity_results = pd.read_csv(root / "similarity/results/similarity_comparison.csv")
    generalization_class = pd.read_csv(root / "generalization/results/classification_generalization.csv").iloc[0]
    generalization_similarity = pd.read_csv(root / "generalization/results/similarity_generalization.csv")
    playlists = pd.read_csv(root / "similarity/results/playlists/playlist_coherence.csv")
    classification_embedding = json.loads((root / "embeddings/results/classification/classification_embedding_metrics.json").read_text())
    triplet_embedding = json.loads((root / "embeddings/results/triplet/triplet_embedding_metrics.json").read_text())

    rows = []
    for model in ("cnn", "gbm"):
        values = genre_raw[genre_raw["model"] == model].pivot(index="fold", columns="augmented", values="accuracy")
        delta, low, high = mean_ci(values[True] - values[False])
        rows.append(_row("augmentation", model, "accuracy_delta", delta, low, high,
                         "genre classification fold-paired CV", "Positive values favor augmentation."))
    for model in ("cnn", "gbm"):
        record = genre_results[(genre_results["model"] == model) & (genre_results["augmented"] == True) & (genre_results["metric"] == "accuracy")].iloc[0]
        rows.append(_row("genre_classification", model, "accuracy", record["mean"], record["ci_low"], record["ci_high"], "genre classification 5-fold CV"))

    for target in ("valence", "arousal"):
        metric = f"{target}_r2"
        for model in ("cnn", "gbm"):
            record = mood[(mood["model"] == model) & (mood["metric"] == metric)].iloc[0]
            rows.append(_row("mood_regression", model, metric, record["mean"], record["ci_low"], record["ci_high"], "mood regression 5-fold CV"))
        pivot = mood_raw.pivot(index="fold", columns="model", values=metric)
        delta, low, high = mean_ci(pivot["gbm"] - pivot["cnn"])
        rows.append(_row("mood_regression", "gbm_minus_cnn", f"{target}_r2_delta", delta, low, high,
                         "mood regression fold-paired CV", "Positive values favor GBM."))

    rows.extend([
        _row("embedding", "classification_embedding", "cosine_silhouette", classification_embedding["embedding_silhouette_cosine"], source="classification embedding"),
        _row("embedding", "triplet_embedding", "cosine_silhouette", triplet_embedding["embedding_silhouette_cosine"], source="triplet embedding training"),
        _row("embedding", "triplet_training", "triplet_loss_reduction",
             triplet_embedding["initial_triplet_loss"] - triplet_embedding["final_triplet_loss"], source="triplet embedding training"),
    ])
    for method in ("classification_embedding", "triplet_embedding", "engineered_features", "metadata_genre"):
        record = similarity_results[(similarity_results["method"] == method) & (similarity_results["k"] == 10)].iloc[0]
        caveat = "Oracle-like label baseline, not audio retrieval." if method == "metadata_genre" else "Genre agreement is a weak similarity proxy."
        rows.append(_row("similarity", method, "precision_at_10", record["precision_at_k"], record["ci_low"], record["ci_high"], "GTZAN in-sample similarity retrieval", caveat))

    rows.append(_row("generalization", "genre_cnn", "fma_accuracy", generalization_class["accuracy"], generalization_class["accuracy_ci_low"], generalization_class["accuracy_ci_high"], "cross-dataset generalization FMA exact-overlap", "Only three exact-overlap genres."))
    rows.append(_row("generalization", "genre_cnn", "absolute_accuracy_drop", generalization_class["absolute_accuracy_drop"], source="GTZAN CV minus FMA"))
    for method in ("classification_embedding", "triplet_embedding"):
        record = generalization_similarity[(generalization_similarity["method"] == method) & (generalization_similarity["k"] == 10)].iloc[0]
        rows.append(_row("generalization", method, "fma_precision_at_10", record["fma_precision_at_k"], record["fma_ci_low"], record["fma_ci_high"], "cross-dataset generalization FMA retrieval"))
        rows.append(_row("generalization", method, "chance_adjusted_precision_drop_at_10", record["adjusted_precision_drop"], source="cross-dataset generalization", caveat="Adjusts for 3 FMA versus 10 GTZAN classes."))

    for method in ("classification_embedding", "triplet_embedding"):
        record = playlists[(playlists["method"] == method) & (playlists["metric"] == "pairwise_similarity")].iloc[0]
        rows.append(_row("playlist", method, "pairwise_similarity_lift_vs_random", record["mean_difference"], record["difference_ci_low"], record["difference_ci_high"], "playlist evaluation 100 seeds", "Compare lifts, not raw cosine levels across spaces."))
    return pd.DataFrame(rows)


def render_report(summary: pd.DataFrame) -> str:
    def value(section, comparison, metric):
        return summary[(summary.section == section) & (summary.comparison == comparison) & (summary.metric == metric)].iloc[0]

    cnn_aug = value("augmentation", "cnn", "accuracy_delta")
    gbm_aug = value("augmentation", "gbm", "accuracy_delta")
    cnn_acc = value("genre_classification", "cnn", "accuracy")
    gbm_acc = value("genre_classification", "gbm", "accuracy")
    cls_p10 = value("similarity", "classification_embedding", "precision_at_10")
    tri_p10 = value("similarity", "triplet_embedding", "precision_at_10")
    fma_acc = value("generalization", "genre_cnn", "fma_accuracy")
    acc_drop = value("generalization", "genre_cnn", "absolute_accuracy_drop")
    cls_ood = value("generalization", "classification_embedding", "chance_adjusted_precision_drop_at_10")
    tri_ood = value("generalization", "triplet_embedding", "chance_adjusted_precision_drop_at_10")
    cls_playlist = value("playlist", "classification_embedding", "pairwise_similarity_lift_vs_random")
    tri_playlist = value("playlist", "triplet_embedding", "pairwise_similarity_lift_vs_random")
    mood_v = value("mood_regression", "gbm", "valence_r2")
    mood_a = value("mood_regression", "gbm", "arousal_r2")

    return f"""# Results and Honest Comparison

## Headline results

| Question | Result |
|---|---|
| Did augmentation help? | CNN accuracy changed by {cnn_aug.value:+.3f} (95% CI {cnn_aug.ci_low:+.3f} to {cnn_aug.ci_high:+.3f}); GBM changed by {gbm_aug.value:+.3f} ({gbm_aug.ci_low:+.3f} to {gbm_aug.ci_high:+.3f}). |
| Best genre classifier | Augmented CNN: {cnn_acc.value:.3f} accuracy versus augmented GBM: {gbm_acc.value:.3f}. |
| Mood regression | GBM reached R²={mood_v.value:.3f} valence and R²={mood_a.value:.3f} arousal; gains over CNN were small. |
| Best in-sample audio retrieval | Classification embedding P@10={cls_p10.value:.3f}; triplet embedding P@10={tri_p10.value:.3f}. |
| Cross-dataset classification | FMA accuracy={fma_acc.value:.3f}, an absolute drop of {acc_drop.value:.3f} from GTZAN CV. |
| More robust embedding OOD | Chance-adjusted P@10 drop: classification={cls_ood.value:.3f}, triplet={tri_ood.value:.3f}. The triplet space degraded less but remained slightly weaker on FMA. |
| Playlist coherence lift over random | Classification={cls_playlist.value:+.3f}; triplet={tri_playlist.value:+.3f}. |

## Honest interpretation

Augmentation produced a substantial observed CNN gain at this dataset size, but the five-fold paired 95% interval narrowly included zero, so the evidence is promising rather than conclusive. Its GBM effect was small and uncertain. Mood remained substantially harder and noisier than genre classification; GBM only narrowly led the CNN.

The purpose-built triplet objective reduced its training loss, but it did **not** beat the simpler classification-derived embedding on genre-based retrieval, silhouette score, or playlist lift. Its positive result was robustness: after correcting for the three-class FMA versus ten-class GTZAN evaluation, its cross-dataset retrieval drop was smaller.

The genre CNN generalized poorly to FMA, confirming a large distribution gap. Similarity evaluation retrieval is in-sample and uses genre agreement as a weak proxy, so its high scores should not be interpreted as human perceptual similarity. The metadata baseline is an oracle-like ceiling because it directly uses the evaluation label. Playlist comparisons are valid as within-space lifts over random; raw cosine values across separately trained spaces are not directly comparable.

## Scope and limitations

- GTZAN is small and contains known duplicates, label faults, and artifacts.
- FMA evaluation used 2,999 tracks from only the three defensible exact-overlap genres.
- DEAM annotations are subjective and include annotator disagreement.
- Similarity and playlist quality still require the generated human-review sheets.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    summary = compile_results(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "comparison.csv", index=False)
    report = render_report(summary)
    (args.output_dir / "findings.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
