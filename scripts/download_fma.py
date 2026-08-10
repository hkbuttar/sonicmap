"""Download a targeted FMA (Free Music Archive) subset for cross-dataset
generalization testing (Step 9), not as a primary training set.

Pulls fma_small (8,000 tracks, 8 balanced genres, 30s clips, ~7.2GB) plus
fma_metadata (track/genre tables) from the official FMA mirror. This is
deliberately the smallest official FMA split, not the full 917GB corpus —
see data/DATASETS.md for why a small, targeted pull is the right call here.

Source: https://github.com/mdeff/fma
"""

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _download_utils import download

BASE_URL = "https://os.unil.cloud.switch.ch/fma"
DEST_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "fma_subset"

ARCHIVES = {
    "fma_metadata.zip": f"{BASE_URL}/fma_metadata.zip",
    "fma_small.zip": f"{BASE_URL}/fma_small.zip",
}


def extract(archive: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)


def main() -> int:
    for name, url in ARCHIVES.items():
        marker = DEST_DIR / (name.replace(".zip", ""))
        if marker.exists():
            print(f"{name} already extracted at {marker}, skipping.")
            continue

        archive_path = DEST_DIR / name
        print(f"Downloading {name} ...")
        download(url, archive_path)

        print(f"Extracting {name} ...")
        extract(archive_path, DEST_DIR)

        if not marker.exists():
            print(f"ERROR: extraction did not produce {marker}", file=sys.stderr)
            return 1

        archive_path.unlink()
        print(f"Done with {name}.")

    n_tracks = sum(1 for _ in (DEST_DIR / "fma_small").rglob("*.mp3"))
    print(f"FMA subset ready: {n_tracks} tracks in {DEST_DIR / 'fma_small'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
