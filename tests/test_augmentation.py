"""Correctness checks for the augmentation transforms, using synthetic
signals with known ground truth (a pure tone's frequency, a fixed SNR)
rather than real audio."""

import numpy as np

from features.config import SAMPLE_RATE
from augmentation.transforms import pitch_shift, time_stretch, add_noise


def _sine(freq: float, sr: int = SAMPLE_RATE, duration: float = 3.0) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _dominant_freq(y: np.ndarray, sr: int) -> float:
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    return float(freqs[np.argmax(spectrum)])


def test_pitch_shift_up_moves_fundamental_frequency():
    base_freq = 440.0
    y = _sine(base_freq)
    shifted = pitch_shift(y, sr=SAMPLE_RATE, n_steps=12.0)  # +1 octave -> 2x freq

    got = _dominant_freq(shifted, SAMPLE_RATE)
    expected = base_freq * 2
    assert abs(got - expected) / expected < 0.03


def test_pitch_shift_down_moves_fundamental_frequency():
    base_freq = 440.0
    y = _sine(base_freq)
    shifted = pitch_shift(y, sr=SAMPLE_RATE, n_steps=-12.0)  # -1 octave -> 0.5x freq

    got = _dominant_freq(shifted, SAMPLE_RATE)
    expected = base_freq * 0.5
    assert abs(got - expected) / expected < 0.03


def test_pitch_shift_preserves_duration():
    y = _sine(440.0)
    shifted = pitch_shift(y, sr=SAMPLE_RATE, n_steps=2.0)
    assert len(shifted) == len(y)


def test_time_stretch_slow_produces_longer_signal():
    y = _sine(440.0)
    stretched = time_stretch(y, sr=SAMPLE_RATE, rate=0.9)
    # librosa time_stretch output length is approximately input_length / rate
    expected_len = len(y) / 0.9
    assert abs(len(stretched) - expected_len) / expected_len < 0.05
    assert len(stretched) > len(y)


def test_time_stretch_fast_produces_shorter_signal():
    y = _sine(440.0)
    stretched = time_stretch(y, sr=SAMPLE_RATE, rate=1.1)
    expected_len = len(y) / 1.1
    assert abs(len(stretched) - expected_len) / expected_len < 0.05
    assert len(stretched) < len(y)


def test_time_stretch_preserves_fundamental_frequency():
    base_freq = 440.0
    y = _sine(base_freq, duration=5.0)
    stretched = time_stretch(y, sr=SAMPLE_RATE, rate=0.9)
    got = _dominant_freq(stretched, SAMPLE_RATE)
    assert abs(got - base_freq) / base_freq < 0.03


def test_add_noise_achieves_approximately_target_snr():
    rng = np.random.default_rng(0)
    y = _sine(440.0, duration=5.0)
    target_snr_db = 20.0
    noisy = add_noise(y, sr=SAMPLE_RATE, snr_db=target_snr_db, rng=rng)

    noise = noisy - y
    signal_power = np.mean(y ** 2)
    noise_power = np.mean(noise ** 2)
    achieved_snr_db = 10 * np.log10(signal_power / noise_power)

    assert abs(achieved_snr_db - target_snr_db) < 1.0


def test_add_noise_preserves_duration_and_is_deterministic_with_seed():
    y = _sine(440.0)
    a = add_noise(y, sr=SAMPLE_RATE, snr_db=20.0, rng=np.random.default_rng(42))
    b = add_noise(y, sr=SAMPLE_RATE, snr_db=20.0, rng=np.random.default_rng(42))
    assert len(a) == len(y)
    np.testing.assert_array_equal(a, b)
