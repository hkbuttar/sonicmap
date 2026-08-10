"""Train and evaluate the mood CNN for one cross-validation fold."""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mood.cnn_model import MoodCNN
from mood.dataset import MoodMelDataset
from mood.metrics import regression_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    spec = json.loads(open(args.spec).read())

    y_train = np.asarray(spec["train_targets"], dtype=np.float32)
    y_val = np.asarray(spec["val_targets"], dtype=np.float32)
    mean = y_train.mean(axis=0)
    scale = y_train.std(axis=0)
    scale[scale == 0] = 1.0
    train_ds = MoodMelDataset(spec["train_mel_paths"], (y_train - mean) / scale)
    val_ds = MoodMelDataset(spec["val_mel_paths"], y_val)

    seed = spec.get("seed", 42)
    torch.manual_seed(seed)
    model = MoodCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loader = DataLoader(train_ds, batch_size=spec.get("batch_size", 64), shuffle=True, num_workers=0)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(spec.get("epochs", 15)):
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

    predictions = []
    model.eval()
    with torch.no_grad():
        for x, _ in DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0):
            predictions.append(model(x).numpy() * scale + mean)
    metrics = regression_metrics(y_val, np.concatenate(predictions))
    with open(args.out, "w") as output:
        json.dump(metrics, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
