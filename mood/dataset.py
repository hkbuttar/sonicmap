"""Lazy dataset for cached DEAM mel-spectrograms and 2D mood targets."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from classification.dataset import DB_FLOOR


class MoodMelDataset(Dataset):
    def __init__(self, mel_paths, targets):
        if len(mel_paths) != len(targets):
            raise ValueError("mel_paths and targets must have equal length")
        self.mel_paths = [Path(path) for path in mel_paths]
        self.targets = np.asarray(targets, dtype=np.float32)

    def __len__(self):
        return len(self.mel_paths)

    def __getitem__(self, idx):
        mel = np.load(self.mel_paths[idx]).astype(np.float32)
        mel = np.clip((mel - DB_FLOOR) / -DB_FLOOR, 0.0, 1.0) * 2.0 - 1.0
        return torch.from_numpy(mel).unsqueeze(0), torch.from_numpy(self.targets[idx])
