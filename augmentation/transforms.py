"""Waveform augmentation transforms for addressing GTZAN's small size:
pitch-shift, time-stretch, and additive Gaussian noise.

Each transform operates on a raw waveform (pre fixed-length trim/pad) and
returns a waveform that may have a different sample count than the input
(time-stretch does; pitch-shift and noise injection don't) — callers are
responsible for re-fixing duration before feature extraction.
"""

import numpy as np
import librosa


def pitch_shift(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps).astype(np.float32)


def time_stretch(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    del sr  # unused; kept so every transform shares the (y, sr, **kwargs) signature
    return librosa.effects.time_stretch(y=y, rate=rate).astype(np.float32)


def add_noise(y: np.ndarray, sr: int, snr_db: float = 20.0, rng: np.random.Generator = None) -> np.ndarray:
    """Add white Gaussian noise at a target signal-to-noise ratio."""
    del sr  # unused; kept so every transform shares the (y, sr, **kwargs) signature
    rng = rng if rng is not None else np.random.default_rng()
    signal_power = np.mean(y ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=y.shape)
    return (y + noise).astype(np.float32)


# name -> (transform, kwargs), applied in build_augmented.py. Pitch-shift
# and time-stretch ranges match the plan spec (±2 semitones, 0.9x-1.1x);
# noise is a single moderate-SNR variant. Every transform takes (y, sr, **kwargs).
VARIANTS = {
    "pitch_up2": (pitch_shift, {"n_steps": 2.0}),
    "pitch_down2": (pitch_shift, {"n_steps": -2.0}),
    "stretch_slow": (time_stretch, {"rate": 0.9}),
    "stretch_fast": (time_stretch, {"rate": 1.1}),
    "noise_snr20": (add_noise, {"snr_db": 20.0}),
}
