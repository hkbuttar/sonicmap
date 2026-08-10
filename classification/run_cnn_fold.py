"""Standalone subprocess entry point: trains+evaluates GenreCNN for one
fold/condition and writes the result as JSON.

Runs in its own process deliberately — importing torch and xgboost in the
same interpreter segfaults/deadlocks on this machine (each bundles its own
OpenMP runtime; see run_experiment.py's module docstring). This process
must never import xgboost.
"""

import argparse
import json
import sys

from classification.dataset import MelDataset
from classification.train_cnn import train_cnn, evaluate_cnn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Path to JSON spec file")
    parser.add_argument("--out", required=True, help="Path to write JSON result")
    args = parser.parse_args()

    spec = json.loads(open(args.spec).read())

    train_ds = MelDataset(spec["train_mel_paths"], spec["train_labels"])
    val_ds = MelDataset(spec["val_mel_paths"], spec["val_labels"])

    model = train_cnn(
        train_ds,
        n_classes=spec["n_classes"],
        epochs=spec.get("epochs", 15),
        seed=spec.get("seed", 42),
    )
    metrics = evaluate_cnn(model, val_ds)

    with open(args.out, "w") as f:
        json.dump(metrics, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
