"""Step 4: genre classification, CNN vs. gradient boosting, with and
without Step 3's augmentation, under 5-fold stratified cross-validation.

Folds are defined over the 999 original GTZAN tracks only (see cv.py for
why). For each fold: CNN(no-aug) and GBM(no-aug) train on the fold's
original tracks; CNN(aug) and GBM(aug) additionally train on the cached
augmented siblings of those same tracks. Every condition is *evaluated*
only on the fold's held-out original tracks — augmented clips never
appear in a validation set.

CNN and GBM training run as separate subprocesses (run_cnn_fold.py /
run_gbm_fold.py) rather than in-process: importing torch and xgboost in
the same interpreter segfaults/deadlocks on this machine (each bundles
its own conflicting OpenMP runtime). This module itself imports neither.

Usage:
    python -m classification.run_experiment \
        --features-dir data/cache/features/gtzan \
        --augmented-dir data/cache/augmented/gtzan \
        --n-folds 5 --epochs 15 --out classification/results/step4_classification.csv
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from classification.labels import LABEL_TO_IDX, GENRES
from classification.cv import make_folds, mean_ci

CNN_MODULE = "classification.run_cnn_fold"
GBM_MODULE = "classification.run_gbm_fold"


def load_original(features_dir: Path):
    manifest = pd.read_parquet(features_dir / "manifest.parquet")
    engineered = pd.read_parquet(features_dir / "engineered_features.parquet")
    manifest = manifest.sort_values("track_id").reset_index(drop=True)
    engineered = engineered.set_index("track_id").loc[manifest["track_id"]].reset_index()

    y = manifest["label"].map(LABEL_TO_IDX).to_numpy()
    mel_paths = [str(features_dir / p) for p in manifest["mel_path"]]
    X_eng = engineered.drop(columns=["track_id"]).to_numpy(dtype=np.float32)
    track_ids = manifest["track_id"].to_numpy()
    return track_ids, mel_paths, X_eng, y


def load_augmented(augmented_dir: Path):
    manifest = pd.read_parquet(augmented_dir / "manifest.parquet")
    y = manifest["label"].map(LABEL_TO_IDX).to_numpy()
    mel_paths = [str(augmented_dir / p) for p in manifest["mel_path"]]
    eng_paths = [str(augmented_dir / p) for p in manifest["engineered_path"]]
    source_ids = manifest["source_track_id"].to_numpy()
    return source_ids, mel_paths, eng_paths, y


def run_subprocess(module: str, spec: dict, tmp_dir: Path, tag: str) -> dict:
    spec_path = tmp_dir / f"{tag}_spec.json"
    out_path = tmp_dir / f"{tag}_result.json"
    spec_path.write_text(json.dumps(spec))

    result = subprocess.run(
        [sys.executable, "-m", module, "--spec", str(spec_path), "--out", str(out_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{module} ({tag}) failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    return json.loads(out_path.read_text())


def run(features_dir: Path, augmented_dir: Path, n_folds: int, epochs: int, seed: int, tmp_dir: Path) -> pd.DataFrame:
    track_ids, mel_paths, X_eng, y = load_original(features_dir)
    aug_source_ids, aug_mel_paths, aug_eng_paths, aug_y = load_augmented(augmented_dir)

    n_classes = len(GENRES)
    folds = make_folds(y, n_splits=n_folds, seed=seed)

    records = []
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        t0 = time.time()
        train_ids = set(track_ids[train_idx])
        aug_mask = np.isin(aug_source_ids, list(train_ids))

        val_mel = [mel_paths[i] for i in val_idx]
        val_labels = [int(v) for v in y[val_idx]]

        # --- CNN, no augmentation ---
        spec = {
            "train_mel_paths": [mel_paths[i] for i in train_idx],
            "train_labels": [int(v) for v in y[train_idx]],
            "val_mel_paths": val_mel, "val_labels": val_labels,
            "n_classes": n_classes, "epochs": epochs, "seed": seed,
        }
        m = run_subprocess(CNN_MODULE, spec, tmp_dir, f"f{fold_i}_cnn_noaug")
        records.append({"fold": fold_i, "model": "cnn", "augmented": False, **m})
        print(f"[fold {fold_i}] cnn no-aug: {m}")

        # --- CNN, with augmentation ---
        aug_mel = [aug_mel_paths[i] for i in np.where(aug_mask)[0]]
        aug_labels = [int(v) for v in aug_y[aug_mask]]
        spec = {
            "train_mel_paths": [mel_paths[i] for i in train_idx] + aug_mel,
            "train_labels": [int(v) for v in y[train_idx]] + aug_labels,
            "val_mel_paths": val_mel, "val_labels": val_labels,
            "n_classes": n_classes, "epochs": epochs, "seed": seed,
        }
        m = run_subprocess(CNN_MODULE, spec, tmp_dir, f"f{fold_i}_cnn_aug")
        records.append({"fold": fold_i, "model": "cnn", "augmented": True, **m})
        print(f"[fold {fold_i}] cnn aug: {m}")

        # --- GBM, no augmentation ---
        gbm_spec = _dump_gbm_arrays(X_eng[train_idx], y[train_idx], X_eng[val_idx], y[val_idx], tmp_dir, f"f{fold_i}_gbm_noaug")
        gbm_spec.update({"n_classes": n_classes, "seed": seed})
        m = run_subprocess(GBM_MODULE, gbm_spec, tmp_dir, f"f{fold_i}_gbm_noaug")
        records.append({"fold": fold_i, "model": "gbm", "augmented": False, **m})
        print(f"[fold {fold_i}] gbm no-aug: {m}")

        # --- GBM, with augmentation ---
        aug_eng = np.stack([np.load(aug_eng_paths[i]) for i in np.where(aug_mask)[0]])
        X_eng_aug = np.concatenate([X_eng[train_idx], aug_eng])
        y_eng_aug = np.concatenate([y[train_idx], aug_y[aug_mask]])
        gbm_spec = _dump_gbm_arrays(X_eng_aug, y_eng_aug, X_eng[val_idx], y[val_idx], tmp_dir, f"f{fold_i}_gbm_aug")
        gbm_spec.update({"n_classes": n_classes, "seed": seed})
        m = run_subprocess(GBM_MODULE, gbm_spec, tmp_dir, f"f{fold_i}_gbm_aug")
        records.append({"fold": fold_i, "model": "gbm", "augmented": True, **m})
        print(f"[fold {fold_i}] gbm aug: {m}")

        print(f"[fold {fold_i}] done in {time.time() - t0:.1f}s")

    return pd.DataFrame(records)


def _dump_gbm_arrays(X_train, y_train, X_val, y_val, tmp_dir: Path, tag: str) -> dict:
    paths = {
        "X_train_path": tmp_dir / f"{tag}_Xtr.npy",
        "y_train_path": tmp_dir / f"{tag}_ytr.npy",
        "X_val_path": tmp_dir / f"{tag}_Xval.npy",
        "y_val_path": tmp_dir / f"{tag}_yval.npy",
    }
    np.save(paths["X_train_path"], X_train)
    np.save(paths["y_train_path"], y_train)
    np.save(paths["X_val_path"], X_val)
    np.save(paths["y_val_path"], y_val)
    return {k: str(v) for k, v in paths.items()}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, augmented), group in df.groupby(["model", "augmented"]):
        for metric in ["accuracy", "f1_macro"]:
            mean, lo, hi = mean_ci(group[metric].values)
            rows.append({"model": model, "augmented": augmented, "metric": metric,
                         "mean": mean, "ci_low": lo, "ci_high": hi, "n_folds": len(group)})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("data/cache/features/gtzan"))
    parser.add_argument("--augmented-dir", type=Path, default=Path("data/cache/augmented/gtzan"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("classification/results/step4_classification.csv"))
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="sonicmap_cv_") as tmp:
        df = run(args.features_dir, args.augmented_dir, args.n_folds, args.epochs, args.seed, Path(tmp))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    summary = summarize(df)
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\n=== Summary (mean [95% CI] across folds) ===")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
