"""Deterministic genre-supervised triplet sampling over cached mel files."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from classification.dataset import DB_FLOOR


class TripletMelDataset(Dataset):
    def __init__(self, mel_paths, labels, source_ids, samples_per_epoch=2000, seed=42):
        if not (len(mel_paths) == len(labels) == len(source_ids)):
            raise ValueError("mel paths, labels, and source IDs must have equal length")
        self.mel_paths = [Path(path) for path in mel_paths]
        self.labels = np.asarray(labels)
        self.source_ids = np.asarray(source_ids)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        self.positive_indices = {}
        self.negative_indices = {}
        for label in np.unique(self.labels):
            self.positive_indices[label] = np.flatnonzero(self.labels == label)
            self.negative_indices[label] = np.flatnonzero(self.labels != label)

    def __len__(self):
        return self.samples_per_epoch

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def sample_indices(self, idx):
        rng = np.random.default_rng(self.seed + self.epoch * self.samples_per_epoch + idx)
        # Draw anchors across the full original+augmented pool. Using idx
        # directly would bias a shorter epoch toward whichever variants
        # happen to appear first in the manifest.
        anchor = int(rng.integers(len(self.mel_paths)))
        label = self.labels[anchor]
        positives = self.positive_indices[label]
        positives = positives[self.source_ids[positives] != self.source_ids[anchor]]
        if not len(positives):
            raise ValueError(f"No distinct positive available for label {label}")
        positive = int(rng.choice(positives))
        negative = int(rng.choice(self.negative_indices[label]))
        return anchor, positive, negative

    def _load(self, idx):
        mel = np.load(self.mel_paths[idx]).astype(np.float32)
        mel = np.clip((mel - DB_FLOOR) / -DB_FLOOR, 0.0, 1.0) * 2.0 - 1.0
        return torch.from_numpy(mel).unsqueeze(0)

    def __getitem__(self, idx):
        anchor, positive, negative = self.sample_indices(idx)
        return self._load(anchor), self._load(positive), self._load(negative)
