"""Load and align DEAM song-level labels with cached audio features."""

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMNS = ("valence", "arousal")


def load_deam_labels(annotations_dir: Path) -> pd.DataFrame:
    """Read DEAM's two song-level annotation files into one clean table."""
    paths = sorted(Path(annotations_dir).glob("static_annotations_averaged_songs_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No DEAM song-level annotations found in {annotations_dir}")

    labels = pd.concat((pd.read_csv(path, skipinitialspace=True) for path in paths), ignore_index=True)
    required = {"song_id", "valence_mean", "arousal_mean"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"DEAM annotations are missing columns: {sorted(missing)}")

    labels = labels[["song_id", "valence_mean", "arousal_mean"]].rename(
        columns={"valence_mean": "valence", "arousal_mean": "arousal"}
    )
    labels["track_id"] = labels.pop("song_id").astype(str)
    labels = labels.dropna().drop_duplicates("track_id", keep=False)
    return labels[["track_id", *TARGET_COLUMNS]].sort_values("track_id").reset_index(drop=True)


def load_deam_dataset(features_dir: Path, annotations_dir: Path):
    """Return aligned track IDs, mel paths, engineered features and targets."""
    features_dir = Path(features_dir)
    manifest = pd.read_parquet(features_dir / "manifest.parquet")
    engineered = pd.read_parquet(features_dir / "engineered_features.parquet")
    labels = load_deam_labels(annotations_dir)

    joined = manifest[["track_id", "mel_path"]].merge(labels, on="track_id", validate="one_to_one")
    joined = joined.merge(engineered, on="track_id", validate="one_to_one").sort_values("track_id")
    if joined.empty:
        raise ValueError("No cached DEAM tracks matched the song-level annotations")

    feature_columns = [c for c in engineered.columns if c != "track_id"]
    track_ids = joined["track_id"].to_numpy()
    mel_paths = [str(features_dir / path) for path in joined["mel_path"]]
    X = joined[feature_columns].to_numpy(dtype=np.float32)
    y = joined[list(TARGET_COLUMNS)].to_numpy(dtype=np.float32)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("DEAM features or targets contain non-finite values")
    return track_ids, mel_paths, X, y
