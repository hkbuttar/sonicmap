"""PyTorch Dataset over cached mel-spectrogram .npy files, loaded lazily
per-item (rather than held in memory) since the full augmented set runs
several GB."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# power_to_db's default top_db=80 puts every clip's dB range at roughly
# [-80, 0]; this rescales to [-1, 1] with a fixed formula (not fit from
# data), so no train/val leakage risk from normalization stats.
DB_FLOOR = -80.0


class MelDataset(Dataset):
    def __init__(self, mel_paths: list, labels: list):
        assert len(mel_paths) == len(labels)
        self.mel_paths = [Path(p) for p in mel_paths]
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.mel_paths)

    def __getitem__(self, idx: int):
        mel = np.load(self.mel_paths[idx]).astype(np.float32)
        mel = np.clip((mel - DB_FLOOR) / -DB_FLOOR, 0.0, 1.0) * 2.0 - 1.0
        x = torch.from_numpy(mel).unsqueeze(0)  # (1, n_mels, T)
        y = int(self.labels[idx])
        return x, y
