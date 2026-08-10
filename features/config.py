"""Shared feature-extraction constants.

Kept in one place because downstream steps (CNN training, embeddings,
cross-dataset generalization) all depend on GTZAN and FMA producing
identically-shaped feature arrays.
"""

SAMPLE_RATE = 22050

# GTZAN clips run 29.93s-30.65s; FMA "small" previews are 30s. 29.0s is a
# floor both datasets clear, so every track is trimmed (never padded) and
# every mel-spectrogram comes out the same shape.
TARGET_DURATION_S = 29.0

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 20
