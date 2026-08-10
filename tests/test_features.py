"""Correctness sanity checks for feature extraction, using synthetic
signals with known properties rather than real audio — a pure tone's
spectral peak location and a silent clip's energy are ground truth we can
assert on exactly, which spot-checks whether the librosa calls in
features/spectral.py and features/engineered.py are wired correctly.
"""

import numpy as np
import librosa

from features.config import SAMPLE_RATE, TARGET_DURATION_S, N_MELS
from features.spectral import compute_mel_spectrogram
from features.engineered import compute_engineered_features


def _sine(freq: float, sr: int = SAMPLE_RATE, duration: float = TARGET_DURATION_S, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_mel_spectrogram_shape():
    y = _sine(440.0)
    mel = compute_mel_spectrogram(y, sr=SAMPLE_RATE)
    expected_frames = 1 + len(y) // 512
    assert mel.shape == (N_MELS, expected_frames)


def test_mel_spectrogram_peaks_near_tone_frequency():
    freq = 440.0
    y = _sine(freq)
    mel = compute_mel_spectrogram(y, sr=SAMPLE_RATE)

    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=SAMPLE_RATE / 2)
    expected_bin = int(np.argmin(np.abs(mel_freqs - freq)))

    energy_per_bin = mel.mean(axis=1)
    loudest_bin = int(np.argmax(energy_per_bin))

    # Mel bins are non-uniformly spaced; allow a small window rather than
    # requiring the exact nearest-bin index.
    assert abs(loudest_bin - expected_bin) <= 2, (
        f"loudest mel bin {loudest_bin} (~{mel_freqs[loudest_bin]:.0f}Hz) "
        f"too far from expected bin {expected_bin} (~{mel_freqs[expected_bin]:.0f}Hz) for a {freq}Hz tone"
    )


def test_higher_frequency_tone_shifts_peak_bin_higher():
    mel_low = compute_mel_spectrogram(_sine(220.0), sr=SAMPLE_RATE)
    mel_high = compute_mel_spectrogram(_sine(3000.0), sr=SAMPLE_RATE)

    peak_low = np.argmax(mel_low.mean(axis=1))
    peak_high = np.argmax(mel_high.mean(axis=1))

    assert peak_high > peak_low


def test_engineered_features_finite_and_fixed_length():
    y = _sine(440.0)
    vector, names = compute_engineered_features(y, sr=SAMPLE_RATE)

    assert vector.shape == (len(names),)
    assert np.all(np.isfinite(vector))
    assert len(names) == len(set(names)), "feature names must be unique"


def test_rms_energy_scales_with_amplitude():
    loud, names = compute_engineered_features(_sine(440.0, amplitude=0.9), sr=SAMPLE_RATE)
    quiet, _ = compute_engineered_features(_sine(440.0, amplitude=0.05), sr=SAMPLE_RATE)

    rms_mean_idx = names.index("rms_0_mean")
    assert loud[rms_mean_idx] > quiet[rms_mean_idx]


def test_silence_has_near_zero_energy():
    y = np.zeros(int(SAMPLE_RATE * TARGET_DURATION_S), dtype=np.float32)
    vector, names = compute_engineered_features(y, sr=SAMPLE_RATE)

    rms_mean_idx = names.index("rms_0_mean")
    assert vector[rms_mean_idx] < 1e-6
