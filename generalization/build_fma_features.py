"""Cache features for the exact-overlap portion of FMA Small (3 genres)."""

import argparse
from pathlib import Path

import pandas as pd

from features.build_features import build
from generalization.taxonomy import FMA_TO_GTZAN, fma_audio_path


def select_tracks(metadata_path, audio_root):
    tracks = pd.read_csv(metadata_path, header=[0, 1], index_col=0)
    small = tracks[tracks[("set", "subset")] == "small"]
    selected = small[small[("track", "genre_top")].isin(FMA_TO_GTZAN)].copy()
    rows = []
    for track_id, row in selected.iterrows():
        path = fma_audio_path(audio_root, track_id)
        if path.exists():
            fma_genre = row[("track", "genre_top")]
            rows.append({
                "fma_track_id": int(track_id), "audio_path": path,
                "fma_genre": fma_genre, "label": FMA_TO_GTZAN[fma_genre],
            })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, default=Path("data/raw/fma_subset/fma_small"))
    parser.add_argument("--metadata", type=Path, default=Path("data/raw/fma_subset/fma_metadata/tracks.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/features/fma_overlap"))
    args = parser.parse_args()
    selected = select_tracks(args.metadata, args.audio_root)
    if selected.empty:
        raise FileNotFoundError("No exact-overlap FMA Small audio tracks matched the metadata")
    manifest, _ = build(args.audio_root, args.cache_dir, False, files=selected["audio_path"])
    lookup = selected.set_index("fma_track_id")
    manifest["fma_track_id"] = manifest["track_id"].map(lambda value: int(Path(value).name))
    manifest["fma_genre"] = manifest["fma_track_id"].map(lookup["fma_genre"])
    manifest["label"] = manifest["fma_track_id"].map(lookup["label"])
    if manifest["label"].isna().any():
        raise ValueError("Cached FMA manifest contains tracks outside the selected taxonomy")
    manifest.to_parquet(args.cache_dir / "manifest.parquet", index=False)
    print(f"FMA overlap cache ready: {len(manifest)} tracks")
    print(manifest["label"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
