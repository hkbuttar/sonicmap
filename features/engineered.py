"""Engineered feature vector for gradient boosting and feature-based
similarity: MFCCs, chroma,
tempo, and spectral-shape statistics, each summarized as mean+std over
time so every track yields a fixed-length vector regardless of frame count.
"""

from typing import Tuple

import numpy as np
import librosa

from features.config import HOP_LENGTH, N_MFCC


def _stats(feat: np.ndarray, name: str) -> Tuple[np.ndarray, list]:
    mean = feat.mean(axis=1)
    std = feat.std(axis=1)
    names = [f"{name}_{i}_mean" for i in range(len(mean))] + [f"{name}_{i}_std" for i in range(len(std))]
    return np.concatenate([mean, std]), names


def compute_engineered_features(y: np.ndarray, sr: int) -> Tuple[np.ndarray, list]:
    values, names = [], []

    def add(feat, name):
        v, n = _stats(feat, name)
        values.append(v)
        names.extend(n)

    add(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH), "mfcc")
    add(librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH), "chroma")
    add(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH), "spectral_centroid")
    add(librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH), "spectral_bandwidth")
    add(librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH), "spectral_rolloff")
    add(librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH), "zcr")
    add(librosa.feature.rms(y=y, hop_length=HOP_LENGTH), "rms")

    tempo = np.atleast_1d(librosa.feature.tempo(y=y, sr=sr, hop_length=HOP_LENGTH))[:1].astype(np.float32)
    values.append(tempo)
    names.append("tempo")

    vector = np.concatenate(values).astype(np.float32)
    return vector, names
