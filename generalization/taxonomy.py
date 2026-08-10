"""Conservative FMA-to-GTZAN genre alignment for Step 9."""

FMA_TO_GTZAN = {
    "Hip-Hop": "hiphop",
    "Pop": "pop",
    "Rock": "rock",
}


def fma_audio_path(audio_root, track_id):
    """Resolve an integer FMA ID using the official zero-padded layout."""
    padded = f"{int(track_id):06d}"
    return audio_root / padded[:3] / f"{padded}.mp3"
