"""Step 9: GTZAN-trained classifier and embeddings evaluated on FMA."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from classification.cnn_model import GenreCNN
from classification.dataset import MelDataset
from classification.labels import GENRES, LABEL_TO_IDX
from embeddings.classification_embedding import extract_embeddings
from embeddings.triplet_model import TripletEmbeddingCNN
from similarity.evaluation import bootstrap_mean_ci, nearest_neighbors, precision_at_k


def load_fma(cache_dir):
    cache_dir = Path(cache_dir)
    manifest = pd.read_parquet(cache_dir / "manifest.parquet").sort_values("track_id").reset_index(drop=True)
    unknown = set(manifest["label"]).difference(LABEL_TO_IDX)
    if unknown:
        raise ValueError(f"Unmapped FMA labels: {sorted(unknown)}")
    paths = [str(cache_dir / path) for path in manifest["mel_path"]]
    labels = manifest["label"].map(LABEL_TO_IDX).to_numpy()
    return manifest, paths, labels


def load_models(classification_checkpoint, triplet_checkpoint):
    classifier_state = torch.load(classification_checkpoint, map_location="cpu", weights_only=True)
    classifier = GenreCNN(n_classes=len(GENRES), embedding_dim=classifier_state["embedding_dim"])
    classifier.load_state_dict(classifier_state["model_state_dict"])
    triplet_state = torch.load(triplet_checkpoint, map_location="cpu", weights_only=True)
    triplet = TripletEmbeddingCNN(embedding_dim=triplet_state["embedding_dim"])
    triplet.load_state_dict(triplet_state["model_state_dict"])
    return classifier, triplet


def classifier_predictions(model, mel_paths, labels):
    loader = DataLoader(MelDataset(mel_paths, labels), batch_size=64, shuffle=False, num_workers=0)
    predictions = []
    model.eval()
    with torch.no_grad():
        for audio, _ in loader:
            predictions.append(model(audio).argmax(dim=1).numpy())
    return np.concatenate(predictions)


def adjusted_precision(precision, random_baseline):
    """Chance-adjust precision so 3-class FMA and 10-class GTZAN compare."""
    return (precision - random_baseline) / (1 - random_baseline)


def per_genre_accuracy(labels, predictions):
    """Summarize accuracy by encoded genre without groupby.apply quirks."""
    frame = pd.DataFrame({
        "label": labels, "correct": np.asarray(labels) == np.asarray(predictions),
    })
    result = frame.groupby("label", as_index=False).agg(
        n=("correct", "size"), accuracy=("correct", "mean"),
    )
    result["genre"] = result["label"].map(dict(enumerate(GENRES)))
    return result


def run(cache_dir, classification_dir, triplet_dir, step4_summary, step8_comparison,
        output_dir, ks, n_bootstrap, seed):
    manifest, mel_paths, labels = load_fma(cache_dir)
    classifier, triplet_model = load_models(
        Path(classification_dir) / "genre_cnn_full.pt",
        Path(triplet_dir) / "triplet_embedding_cnn.pt",
    )
    predictions = classifier_predictions(classifier, mel_paths, labels)
    correct = (predictions == labels).astype(float)
    accuracy, acc_low, acc_high = bootstrap_mean_ci(correct, n_bootstrap=n_bootstrap, seed=seed)
    fma_classification = {
        "dataset": "fma_exact_overlap", "n_tracks": len(labels), "n_genres": len(np.unique(labels)),
        "accuracy": accuracy, "accuracy_ci_low": acc_low, "accuracy_ci_high": acc_high,
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
    }

    step4 = pd.read_csv(step4_summary)
    in_dist_acc = float(step4[(step4["model"] == "cnn") & (step4["augmented"].astype(str).str.lower() == "true") & (step4["metric"] == "accuracy")]["mean"].iloc[0])
    fma_classification["gtzan_cv_accuracy"] = in_dist_acc
    fma_classification["absolute_accuracy_drop"] = in_dist_acc - accuracy

    classification_embeddings = extract_embeddings(classifier, mel_paths, labels)
    triplet_embeddings = extract_embeddings(triplet_model, mel_paths, labels)
    step8 = pd.read_csv(step8_comparison)
    random_fma = float(np.sum((np.bincount(labels) / len(labels)) ** 2))
    random_gtzan = 0.1
    retrieval_rows = []
    neighbor_tables = []
    for model_idx, (name, vectors) in enumerate((
        ("classification_embedding", classification_embeddings),
        ("triplet_embedding", triplet_embeddings),
    )):
        indices, distances = nearest_neighbors(vectors, metric="cosine", max_k=max(ks))
        for query_idx in range(len(labels)):
            for rank, neighbor_idx in enumerate(indices[query_idx], start=1):
                neighbor_tables.append({
                    "method": name, "query_track_id": manifest.loc[query_idx, "fma_track_id"],
                    "query_label": manifest.loc[query_idx, "label"], "rank": rank,
                    "neighbor_track_id": manifest.loc[neighbor_idx, "fma_track_id"],
                    "neighbor_label": manifest.loc[neighbor_idx, "label"],
                    "distance": float(distances[query_idx, rank - 1]),
                })
        for k in ks:
            values = precision_at_k(indices, labels, k)
            mean, low, high = bootstrap_mean_ci(values, n_bootstrap=n_bootstrap, seed=seed + model_idx * 100 + k)
            gtzan_precision = float(step8[(step8["method"] == name) & (step8["k"] == k)]["precision_at_k"].iloc[0])
            fma_adjusted = adjusted_precision(mean, random_fma)
            gtzan_adjusted = adjusted_precision(gtzan_precision, random_gtzan)
            retrieval_rows.append({
                "method": name, "k": k, "fma_precision_at_k": mean,
                "fma_ci_low": low, "fma_ci_high": high,
                "gtzan_precision_at_k": gtzan_precision,
                "raw_precision_drop": gtzan_precision - mean,
                "fma_random_baseline": random_fma, "gtzan_random_baseline": random_gtzan,
                "fma_adjusted_precision": fma_adjusted,
                "gtzan_adjusted_precision": gtzan_adjusted,
                "adjusted_precision_drop": gtzan_adjusted - fma_adjusted,
            })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([fma_classification]).to_csv(output_dir / "step9_classification_generalization.csv", index=False)
    pd.DataFrame(retrieval_rows).to_csv(output_dir / "step9_similarity_generalization.csv", index=False)
    pd.DataFrame(neighbor_tables).to_parquet(output_dir / "step9_fma_neighbors.parquet", index=False)
    per_class = per_genre_accuracy(labels, predictions)
    per_class.to_csv(output_dir / "step9_per_genre_accuracy.csv", index=False)
    protocol = {
        "training_dataset": "GTZAN", "evaluation_dataset": "FMA Small exact taxonomy overlap",
        "taxonomy_mapping": {"Hip-Hop": "hiphop", "Pop": "pop", "Rock": "rock"},
        "excluded_fma_genres": ["Electronic", "Experimental", "Folk", "Instrumental", "International"],
        "reason": "Excluded genres have no defensible exact GTZAN equivalent.",
        "retrieval_note": "Chance-adjusted precision is the primary cross-dataset comparison because FMA evaluation has 3 classes and GTZAN has 10.",
    }
    (output_dir / "step9_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return fma_classification, pd.DataFrame(retrieval_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/features/fma_overlap"))
    parser.add_argument("--classification-dir", type=Path, default=Path("embeddings/results/classification"))
    parser.add_argument("--triplet-dir", type=Path, default=Path("embeddings/results/triplet"))
    parser.add_argument("--step4-summary", type=Path, default=Path("classification/results/step4_classification_summary.csv"))
    parser.add_argument("--step8-comparison", type=Path, default=Path("similarity/results/step8_similarity_comparison.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("generalization/results"))
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        classification, retrieval = run(
            args.cache_dir, args.classification_dir, args.triplet_dir,
            args.step4_summary, args.step8_comparison, args.output_dir,
            tuple(sorted(set(args.ks))), args.bootstrap_samples, args.seed,
        )
    except FileNotFoundError as error:
        if not (args.cache_dir / "manifest.parquet").exists():
            parser.error(f"{error}. First run: .venv/bin/python -m generalization.build_fma_features")
        raise
    print("\n=== FMA classification generalization ===")
    print(pd.DataFrame([classification]).to_string(index=False))
    print("\n=== FMA similarity generalization ===")
    print(retrieval.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
