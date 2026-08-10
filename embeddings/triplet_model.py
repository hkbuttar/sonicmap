"""CNN trained directly as a normalized audio-similarity embedding."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletEmbeddingCNN(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2, 4)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 4)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, embedding_dim), nn.ReLU(),
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.embedding(self.conv(x)), p=2, dim=1)

    def forward(self, anchor, positive, negative):
        return self.embed(anchor), self.embed(positive), self.embed(negative)
