"""Compact CNN over log-mel spectrograms. Kept small deliberately: this
trains from scratch, CPU-only, on a few hundred to a few thousand 29s
clips per fold, so a small parameter count matters for wall-clock time,
not just overfitting risk.

`embed()` exposes the 128-dim penultimate layer directly; embedding extraction reuses
this as the classification-derived embedding, extracted from a model
trained on the full dataset rather than retrained from scratch.
"""

import torch
import torch.nn as nn


class GenreCNN(nn.Module):
    def __init__(self, n_classes: int = 10, embedding_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2, 4)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 4)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.embed_fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, embedding_dim), nn.ReLU())
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed_fc(self.conv(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.embed(x)))
