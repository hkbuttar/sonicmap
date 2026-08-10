"""Audio loading at a fixed sample rate and duration, so every track
(regardless of source dataset or exact clip length) produces feature
arrays of identical shape."""

import numpy as np
import librosa

from features.config import SAMPLE_RATE, TARGET_DURATION_S


def load_audio(path, sr: int = SAMPLE_RATE, duration: float = TARGET_DURATION_S) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    target_len = int(round(sr * duration))
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y.astype(np.float32)
