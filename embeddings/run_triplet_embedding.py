"""Step 7: train and evaluate a purpose-built triplet-loss embedding."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score

from classification.labels import GENRES
from classification.run_experiment import load_augmented, load_original
from embeddings.classification_embedding import extract_embeddings, genre_silhouette, project_2d
from embeddings.train_triplet import train_triplet
from embeddings.triplet_dataset import TripletMelDataset


def run(features_dir, augmented_dir, output_dir, classification_dir, epochs, triplets_per_epoch,
        margin, seed, projection_method, use_augmentation=True):
    track_ids, original_mels, _, labels = load_original(features_dir)
    train_mels = list(original_mels)
    train_labels = labels.tolist()
    source_ids = track_ids.tolist()
    if use_augmentation:
        augmented_sources, augmented_mels, _, augmented_labels = load_augmented(augmented_dir)
        train_mels.extend(augmented_mels)
        train_labels.extend(augmented_labels.tolist())
        source_ids.extend(augmented_sources.tolist())

    dataset = TripletMelDataset(
        train_mels, train_labels, source_ids,
        samples_per_epoch=triplets_per_epoch, seed=seed,
    )
    model, history = train_triplet(dataset, epochs=epochs, margin=margin, seed=seed)
    embeddings = extract_embeddings(model, original_mels, labels)
    projection = project_2d(embeddings, method=projection_method, seed=seed)
    embedding_silhouette = genre_silhouette(embeddings, labels)
    metrics = {
        "embedding_silhouette_cosine": embedding_silhouette,
        "projection_silhouette_euclidean": float(silhouette_score(projection, labels)),
        "initial_triplet_loss": history[0]["triplet_loss"] if history else None,
        "final_triplet_loss": history[-1]["triplet_loss"] if history else None,
        "n_tracks": int(len(track_ids)), "embedding_dim": int(embeddings.shape[1]),
        "training_samples": int(len(train_mels)), "triplets_per_epoch": int(triplets_per_epoch),
        "augmented_training": bool(use_augmentation), "margin": float(margin),
        "projection_method": projection_method, "epochs": int(epochs), "seed": int(seed),
    }
    baseline_path = Path(classification_dir) / "classification_embedding_metrics.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())["embedding_silhouette_cosine"]
        metrics["classification_embedding_silhouette_cosine"] = float(baseline)
        metrics["silhouette_delta_vs_classification"] = float(embedding_silhouette - baseline)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "triplet_embeddings.npy", embeddings)
    projection_path = output_dir / "triplet_projection.csv"
    pd.DataFrame({
        "track_id": track_ids, "label": [GENRES[index] for index in labels],
        "label_index": labels, "x": projection[:, 0], "y": projection[:, 1],
    }).to_csv(projection_path, index=False)
    pd.DataFrame(history).to_csv(output_dir / "triplet_training_history.csv", index=False)
    (output_dir / "triplet_embedding_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save({
        "model_state_dict": model.state_dict(), "genres": GENRES,
        "embedding_dim": embeddings.shape[1], "margin": margin, "seed": seed,
    }, output_dir / "triplet_embedding_cnn.pt")

    with tempfile.TemporaryDirectory(prefix="sonicmap_mpl_") as mpl_config:
        environment = os.environ.copy()
        environment["MPLCONFIGDIR"] = mpl_config
        plot_result = subprocess.run([
            sys.executable, "-m", "embeddings.plot_projection",
            "--projection", str(projection_path),
            "--out", str(output_dir / "triplet_embedding_projection.png"),
            "--method", projection_method, "--title", "Triplet-loss embeddings",
        ], capture_output=True, text=True, env=environment)
    if plot_result.returncode:
        raise RuntimeError(f"Projection plot failed:\n{plot_result.stderr}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("data/cache/features/gtzan"))
    parser.add_argument("--augmented-dir", type=Path, default=Path("data/cache/augmented/gtzan"))
    parser.add_argument("--output-dir", type=Path, default=Path("embeddings/results/triplet"))
    parser.add_argument("--classification-dir", type=Path, default=Path("embeddings/results/classification"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--triplets-per-epoch", type=int, default=2000)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--projection", choices=("umap", "tsne"), default="umap")
    parser.add_argument("--no-augmentation", action="store_true")
    args = parser.parse_args()
    metrics = run(
        args.features_dir, args.augmented_dir, args.output_dir, args.classification_dir,
        args.epochs, args.triplets_per_epoch, args.margin, args.seed,
        args.projection, not args.no_augmentation,
    )
    print("\n=== Triplet-loss embedding ===")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
