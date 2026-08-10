"""Extract and cache mel-spectrograms + engineered features for a dataset
directory of audio files.

Caches per-track to data/cache/features/<dataset>/{mel,engineered}/, so a
re-run only computes tracks that are missing (interrupted runs resume
cheaply; corrupt tracks are logged to errors.csv rather than aborting).

Usage:
    python -m features.build_features \
        --audio-root data/raw/gtzan/genres \
        --cache-dir data/cache/features/gtzan \
        --label-from-parent
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from features.audio_io import load_audio
from features.spectral import compute_mel_spectrogram
from features.engineered import compute_engineered_features
from features.config import SAMPLE_RATE

AUDIO_EXTENSIONS = (".wav", ".mp3", ".au")


def find_audio_files(audio_root: Path) -> list:
    return sorted(f for f in audio_root.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS)


def build(audio_root: Path, cache_dir: Path, label_from_parent: bool, files=None) -> tuple:
    audio_root = Path(audio_root)
    cache_dir = Path(cache_dir)
    mel_dir = cache_dir / "mel"
    engineered_dir = cache_dir / "engineered"
    mel_dir.mkdir(parents=True, exist_ok=True)
    engineered_dir.mkdir(parents=True, exist_ok=True)

    files = find_audio_files(audio_root) if files is None else sorted(Path(f) for f in files)
    if not files:
        raise FileNotFoundError(f"No audio files found under {audio_root}")

    names_path = cache_dir / "engineered_feature_names.json"
    feature_names = json.loads(names_path.read_text()) if names_path.exists() else None

    errors = []
    for f in tqdm(files, desc=f"extracting features ({audio_root.name})"):
        relpath = f.relative_to(audio_root)
        track_id = relpath.with_suffix("").as_posix()
        mel_path = mel_dir / f"{track_id}.npy"
        eng_path = engineered_dir / f"{track_id}.npy"

        if mel_path.exists() and eng_path.exists():
            continue

        try:
            y = load_audio(f, sr=SAMPLE_RATE)
            if not mel_path.exists():
                mel = compute_mel_spectrogram(y, sr=SAMPLE_RATE)
                mel_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(mel_path, mel)
            if not eng_path.exists():
                vector, names = compute_engineered_features(y, sr=SAMPLE_RATE)
                eng_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(eng_path, vector)
                if feature_names is None:
                    feature_names = names
                    names_path.write_text(json.dumps(feature_names))
        except Exception as e:
            errors.append({"track_id": track_id, "path": str(f), "error": f"{type(e).__name__}: {e}"})

    if errors:
        pd.DataFrame(errors).to_csv(cache_dir / "errors.csv", index=False)

    manifest, engineered_df = assemble_manifest(audio_root, cache_dir, label_from_parent, feature_names)
    print(
        f"Done. {len(manifest)} tracks cached, {len(errors)} errors "
        f"(see {cache_dir / 'errors.csv'})." if errors else
        f"Done. {len(manifest)} tracks cached, 0 errors."
    )
    return manifest, engineered_df


def assemble_manifest(audio_root: Path, cache_dir: Path, label_from_parent: bool, feature_names) -> tuple:
    """Rebuild manifest.parquet / engineered_features.parquet from whatever
    is currently on disk in the cache — idempotent, so safe to call after
    a partial run."""
    mel_dir = cache_dir / "mel"
    engineered_dir = cache_dir / "engineered"

    track_ids = sorted(p.relative_to(mel_dir).with_suffix("").as_posix() for p in mel_dir.rglob("*.npy"))
    rows, engineered_rows = [], []
    for track_id in track_ids:
        eng_path = engineered_dir / f"{track_id}.npy"
        if not eng_path.exists():
            continue
        relpath = Path(track_id)
        label = relpath.parts[0] if label_from_parent else None
        rows.append({
            "track_id": track_id,
            "label": label,
            "mel_path": str((mel_dir / f"{track_id}.npy").relative_to(cache_dir)),
            "engineered_path": str(eng_path.relative_to(cache_dir)),
        })
        engineered_rows.append(np.load(eng_path))

    manifest = pd.DataFrame(rows)
    if feature_names and engineered_rows:
        engineered_df = pd.DataFrame(np.stack(engineered_rows), columns=feature_names)
        engineered_df.insert(0, "track_id", manifest["track_id"].values)
    else:
        engineered_df = pd.DataFrame()

    manifest.to_parquet(cache_dir / "manifest.parquet", index=False)
    if not engineered_df.empty:
        engineered_df.to_parquet(cache_dir / "engineered_features.parquet", index=False)
    return manifest, engineered_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--label-from-parent", action="store_true",
                         help="Use each file's immediate parent directory name as its genre label (GTZAN layout).")
    args = parser.parse_args()

    build(args.audio_root, args.cache_dir, args.label_from_parent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
