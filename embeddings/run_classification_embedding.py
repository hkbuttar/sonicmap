"""Step 6: train the full genre CNN and extract its penultimate embedding.

The augmented condition won Step 4's genre-CNN comparison, so augmentation
is enabled by default for training. Embeddings and projections are always
extracted for original GTZAN tracks only.
"""

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

from classification.dataset import MelDataset
from classification.labels import GENRES
from classification.run_experiment import load_augmented, load_original
from classification.train_cnn import train_cnn
from embeddings.classification_embedding import (
    extract_embeddings, genre_silhouette, l2_normalize, project_2d,
)


def run(features_dir, augmented_dir, output_dir, epochs, seed, projection_method, use_augmentation=True):
    track_ids, original_mels, _, labels = load_original(features_dir)
    train_mels = list(original_mels)
    train_labels = labels.tolist()
    if use_augmentation:
        _, augmented_mels, _, augmented_labels = load_augmented(augmented_dir)
        train_mels.extend(augmented_mels)
        train_labels.extend(augmented_labels.tolist())

    model = train_cnn(
        MelDataset(train_mels, train_labels), n_classes=len(GENRES),
        epochs=epochs, seed=seed,
    )
    embeddings = extract_embeddings(model, original_mels, labels)
    normalized = l2_normalize(embeddings)
    projection = project_2d(normalized, method=projection_method, seed=seed)
    metrics = {
        "embedding_silhouette_cosine": genre_silhouette(normalized, labels),
        "projection_silhouette_euclidean": float(silhouette_score(projection, labels)),
        "n_tracks": int(len(track_ids)),
        "embedding_dim": int(embeddings.shape[1]),
        "training_samples": int(len(train_mels)),
        "augmented_training": bool(use_augmentation),
        "projection_method": projection_method,
        "epochs": int(epochs),
        "seed": int(seed),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "classification_embeddings.npy", normalized)
    projection_path = output_dir / "classification_projection.csv"
    pd.DataFrame({
        "track_id": track_ids,
        "label": [GENRES[index] for index in labels],
        "label_index": labels,
        "x": projection[:, 0],
        "y": projection[:, 1],
    }).to_csv(projection_path, index=False)
    (output_dir / "classification_embedding_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save({
        "model_state_dict": model.state_dict(), "genres": GENRES,
        "embedding_dim": embeddings.shape[1], "seed": seed,
    }, output_dir / "genre_cnn_full.pt")
    with tempfile.TemporaryDirectory(prefix="sonicmap_mpl_") as mpl_config:
        environment = os.environ.copy()
        environment["MPLCONFIGDIR"] = mpl_config
        plot_result = subprocess.run([
            sys.executable, "-m", "embeddings.plot_projection",
            "--projection", str(projection_path),
            "--out", str(output_dir / "classification_embedding_projection.png"),
            "--method", projection_method,
        ], capture_output=True, text=True, env=environment)
    if plot_result.returncode:
        raise RuntimeError(f"Projection plot failed:\n{plot_result.stderr}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("data/cache/features/gtzan"))
    parser.add_argument("--augmented-dir", type=Path, default=Path("data/cache/augmented/gtzan"))
    parser.add_argument("--output-dir", type=Path, default=Path("embeddings/results/classification"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--projection", choices=("umap", "tsne"), default="umap")
    parser.add_argument("--no-augmentation", action="store_true")
    args = parser.parse_args()
    metrics = run(
        args.features_dir, args.augmented_dir, args.output_dir, args.epochs,
        args.seed, args.projection, not args.no_augmentation,
    )
    print("\n=== Classification-derived embedding ===")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
