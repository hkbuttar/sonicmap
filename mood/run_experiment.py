"""Compare CNN and GBM valence-arousal regression with CV confidence intervals."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from classification.cv import mean_ci
from mood.data import load_deam_dataset

METRICS = tuple(f"{target}_{metric}" for target in ("valence", "arousal") for metric in ("mae", "rmse", "r2"))


def _run(module, spec, temp_dir, tag):
    spec_path, out_path = temp_dir / f"{tag}_spec.json", temp_dir / f"{tag}_result.json"
    spec_path.write_text(json.dumps(spec))
    result = subprocess.run([sys.executable, "-m", module, "--spec", str(spec_path), "--out", str(out_path)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{module} failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    return json.loads(out_path.read_text())


def _array_spec(X_train, y_train, X_val, y_val, temp_dir, tag):
    values = {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val}
    spec = {}
    for name, value in values.items():
        path = temp_dir / f"{tag}_{name}.npy"
        np.save(path, value)
        spec[f"{name}_path"] = str(path)
    return spec


def run(features_dir, annotations_dir, n_folds, epochs, seed, temp_dir):
    _, mel_paths, X, y = load_deam_dataset(features_dir, annotations_dir)
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X)
    records = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        cnn_spec = {
            "train_mel_paths": [mel_paths[i] for i in train_idx],
            "val_mel_paths": [mel_paths[i] for i in val_idx],
            "train_targets": y[train_idx].tolist(), "val_targets": y[val_idx].tolist(),
            "epochs": epochs, "seed": seed,
        }
        cnn_metrics = _run("mood.run_cnn_fold", cnn_spec, temp_dir, f"fold{fold}_cnn")
        records.append({"fold": fold, "model": "cnn", **cnn_metrics})
        print(f"[fold {fold}] cnn: {cnn_metrics}")

        gbm_spec = _array_spec(X[train_idx], y[train_idx], X[val_idx], y[val_idx], temp_dir, f"fold{fold}_gbm")
        gbm_spec["seed"] = seed
        gbm_metrics = _run("mood.run_gbm_fold", gbm_spec, temp_dir, f"fold{fold}_gbm")
        records.append({"fold": fold, "model": "gbm", **gbm_metrics})
        print(f"[fold {fold}] gbm: {gbm_metrics}")
    return pd.DataFrame(records)


def summarize(results):
    rows = []
    for model, group in results.groupby("model"):
        for metric in METRICS:
            mean, low, high = mean_ci(group[metric])
            rows.append({"model": model, "metric": metric, "mean": mean, "ci_low": low, "ci_high": high, "n_folds": len(group)})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("data/cache/features/deam"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("data/raw/mood/annotations/annotations averaged per song/song_level"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("mood/results/mood_regression.csv"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="sonicmap_mood_cv_") as temp:
        results = run(args.features_dir, args.annotations_dir, args.n_folds, args.epochs, args.seed, Path(temp))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    summary = summarize(results)
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    print("\n=== Mood regression summary (mean [95% CI] across folds) ===")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
