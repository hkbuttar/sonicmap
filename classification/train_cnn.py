"""Training/eval loop for GenreCNN. CPU-only (torch built against the CPU
wheel index — see requirements.txt), so kept simple: fixed epoch count,
no mixed precision, no LR scheduling."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from classification.cnn_model import GenreCNN


def train_cnn(
    train_ds, n_classes: int, epochs: int = 15, batch_size: int = 64,
    lr: float = 2e-3, weight_decay: float = 1e-4, seed: int = 42, num_workers: int = 4,
) -> GenreCNN:
    torch.manual_seed(seed)
    model = GenreCNN(n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=num_workers > 0)

    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
    return model


def evaluate_cnn(model: GenreCNN, val_ds, batch_size: int = 64, num_workers: int = 4) -> dict:
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            preds.append(logits.argmax(dim=1).numpy())
            targets.append(y.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return {
        "accuracy": accuracy_score(targets, preds),
        "f1_macro": f1_score(targets, preds, average="macro"),
    }
