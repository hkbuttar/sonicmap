"""Log-mel spectrogram extraction used by genre classification, mood
regression, and both learned embeddings."""

import numpy as np
import librosa

from features.config import N_FFT, HOP_LENGTH, N_MELS


def compute_mel_spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)
