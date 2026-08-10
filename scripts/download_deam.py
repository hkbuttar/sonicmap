"""Download DEAM (MediaEval Database for Emotional Analysis in Music):
1,802 excerpts with real continuous valence-arousal annotations, used for
Step 5's mood regression instead of a genre-derived mood proxy.

Source: https://cvml.unige.ch/databases/DEAM/
"""

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _download_utils import download

BASE_URL = "https://cvml.unige.ch/databases/DEAM"
DEST_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "mood"

ARCHIVES = {
    "DEAM_audio.zip": f"{BASE_URL}/DEAM_audio.zip",
    "DEAM_Annotations.zip": f"{BASE_URL}/DEAM_Annotations.zip",
}


def extract(archive: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)


def main() -> int:
    for name, url in ARCHIVES.items():
        archive_path = DEST_DIR / name
        stamp = DEST_DIR / f".{name}.done"
        if stamp.exists():
            print(f"{name} already extracted, skipping.")
            continue

        print(f"Downloading {name} ...")
        download(url, archive_path)

        print(f"Extracting {name} ...")
        extract(archive_path, DEST_DIR)

        archive_path.unlink()
        stamp.touch()
        print(f"Done with {name}.")

    n_audio = sum(1 for _ in DEST_DIR.rglob("*.mp3"))
    print(f"DEAM ready: {n_audio} audio files in {DEST_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
