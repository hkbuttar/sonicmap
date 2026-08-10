"""Generate augmented waveforms for every track in a dataset and cache
their mel-spectrograms + engineered features, one subdirectory per
augmentation variant.

Each cached track keeps its original track_id in the manifest's
`source_track_id` column — Step 4's cross-validation must group folds by
this column (not by the augmented row's own id) so a track and its
augmented siblings never land on opposite sides of a train/test split;
otherwise held-out accuracy would be inflated by near-duplicate leakage.

Mel-spectrograms are stored as float16 (not float32) here specifically to
keep the ~5x larger augmented cache within disk budget; precision loss is
irrelevant for CNN training input.

Usage:
    python -m augmentation.build_augmented \
        --audio-root data/raw/gtzan/genres \
        --cache-dir data/cache/augmented/gtzan \
        --label-from-parent
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from features.spectral import compute_mel_spectrogram
from features.engineered import compute_engineered_features
from features.config import SAMPLE_RATE, TARGET_DURATION_S
from augmentation.transforms import VARIANTS
from features.build_features import find_audio_files

TARGET_LEN = int(round(SAMPLE_RATE * TARGET_DURATION_S))


def _load_full(path, sr: int) -> np.ndarray:
    """Load the clip's full natural length (not fixed/padded) so
    time-stretch has real audio to work with rather than silence padding —
    GTZAN clips run 29.9-30.6s, barely above the 29s target, and
    stretch_fast (rate 1.1) would otherwise compress padding into content."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def _fix_length(y: np.ndarray) -> np.ndarray:
    if len(y) < TARGET_LEN:
        return np.pad(y, (0, TARGET_LEN - len(y)))
    return y[:TARGET_LEN]


def build(audio_root: Path, cache_dir: Path, label_from_parent: bool) -> pd.DataFrame:
    audio_root = Path(audio_root)
    cache_dir = Path(cache_dir)

    files = find_audio_files(audio_root)
    if not files:
        raise FileNotFoundError(f"No audio files found under {audio_root}")

    names_path = cache_dir / "engineered_feature_names.json"
    feature_names = json.loads(names_path.read_text()) if names_path.exists() else None

    errors = []
    total = len(files) * len(VARIANTS)
    with tqdm(total=total, desc=f"augmenting ({audio_root.name})") as bar:
        for f in files:
            relpath = f.relative_to(audio_root)
            source_track_id = relpath.with_suffix("").as_posix()

            try:
                y_raw = _load_full(f, sr=SAMPLE_RATE)
            except Exception as e:
                for variant in VARIANTS:
                    errors.append({"track_id": f"{source_track_id}__{variant}", "source_track_id": source_track_id,
                                    "path": str(f), "error": f"{type(e).__name__}: {e}"})
                bar.update(len(VARIANTS))
                continue

            for variant, (fn, kwargs) in VARIANTS.items():
                track_id = f"{source_track_id}__{variant}"
                mel_path = cache_dir / variant / "mel" / f"{source_track_id}.npy"
                eng_path = cache_dir / variant / "engineered" / f"{source_track_id}.npy"

                if mel_path.exists() and eng_path.exists():
                    bar.update(1)
                    continue

                try:
                    y_aug = fn(y_raw, sr=SAMPLE_RATE, **kwargs)
                    y_aug = _fix_length(y_aug)

                    if not mel_path.exists():
                        mel = compute_mel_spectrogram(y_aug, sr=SAMPLE_RATE).astype(np.float16)
                        mel_path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(mel_path, mel)
                    if not eng_path.exists():
                        vector, names = compute_engineered_features(y_aug, sr=SAMPLE_RATE)
                        eng_path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(eng_path, vector)
                        if feature_names is None:
                            feature_names = names
                            names_path.write_text(json.dumps(feature_names))
                except Exception as e:
                    errors.append({"track_id": track_id, "source_track_id": source_track_id,
                                    "path": str(f), "error": f"{type(e).__name__}: {e}"})
                bar.update(1)

    if errors:
        pd.DataFrame(errors).to_csv(cache_dir / "errors.csv", index=False)

    manifest = assemble_manifest(cache_dir, label_from_parent)
    print(
        f"Done. {len(manifest)} augmented tracks cached across {len(VARIANTS)} variants, "
        f"{len(errors)} errors" + (f" (see {cache_dir / 'errors.csv'})." if errors else ".")
    )
    return manifest


def assemble_manifest(cache_dir: Path, label_from_parent: bool) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        mel_dir = cache_dir / variant / "mel"
        eng_dir = cache_dir / variant / "engineered"
        if not mel_dir.exists():
            continue
        for mel_path in sorted(mel_dir.rglob("*.npy")):
            source_track_id = mel_path.relative_to(mel_dir).with_suffix("").as_posix()
            eng_path = eng_dir / f"{source_track_id}.npy"
            if not eng_path.exists():
                continue
            label = Path(source_track_id).parts[0] if label_from_parent else None
            rows.append({
                "track_id": f"{source_track_id}__{variant}",
                "source_track_id": source_track_id,
                "variant": variant,
                "label": label,
                "mel_path": str(mel_path.relative_to(cache_dir)),
                "engineered_path": str(eng_path.relative_to(cache_dir)),
            })
    manifest = pd.DataFrame(rows)
    manifest.to_parquet(cache_dir / "manifest.parquet", index=False)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--label-from-parent", action="store_true")
    args = parser.parse_args()

    build(args.audio_root, args.cache_dir, args.label_from_parent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
