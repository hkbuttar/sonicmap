"""Download the GTZAN genre dataset (1000 30s clips, 10 genres).

The canonical host (opihi.cs.uvic.ca / marsyas.info) has been unreachable for
years; this pulls the same archive re-hosted on the Hugging Face Hub
(marsyas/gtzan), which mirrors the original files byte-for-byte.

Known dataset issues (Sturm, 2013, "The GTZAN dataset: Its contents, its
faults, their effects on evaluation, and its future use") are documented in
data/DATASETS.md — this script does not attempt to filter or fix them.
"""

import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _download_utils import download

ARCHIVE_URL = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"
DEST_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "gtzan"
ARCHIVE_PATH = DEST_DIR / "genres.tar.gz"
EXPECTED_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]


def extract(archive: Path, dest_dir: Path) -> None:
    with tarfile.open(archive) as tf:
        tf.extractall(dest_dir, filter="data")
    # The archive was built on macOS and carries AppleDouble sidecar files
    # (._*) alongside every real track; they aren't audio, so drop them.
    for junk in dest_dir.rglob("._*"):
        junk.unlink()


def already_extracted(dest_dir: Path) -> bool:
    genres_dir = dest_dir / "genres"
    if not genres_dir.exists():
        return False
    return all((genres_dir / g).is_dir() for g in EXPECTED_GENRES)


def main() -> int:
    if already_extracted(DEST_DIR):
        print(f"GTZAN already present at {DEST_DIR / 'genres'}, skipping.")
        return 0

    print(f"Downloading GTZAN archive to {ARCHIVE_PATH} ...")
    download(ARCHIVE_URL, ARCHIVE_PATH)

    print("Extracting ...")
    extract(ARCHIVE_PATH, DEST_DIR)

    if not already_extracted(DEST_DIR):
        print("ERROR: extraction did not produce the expected genre folders.", file=sys.stderr)
        return 1

    ARCHIVE_PATH.unlink()
    n_tracks = sum(1 for _ in (DEST_DIR / "genres").rglob("*.wav"))
    print(f"Done. {n_tracks} tracks across {len(EXPECTED_GENRES)} genres in {DEST_DIR / 'genres'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
