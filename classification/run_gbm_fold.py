"""Standalone subprocess entry point: trains+evaluates the GBM baseline
for one fold/condition and writes the result as JSON.

Runs in its own process deliberately — see run_cnn_fold.py's docstring.
This process must never import torch.
"""

import argparse
import json
import sys

import numpy as np

from classification.gbm_model import train_gbm, evaluate_gbm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Path to JSON spec file")
    parser.add_argument("--out", required=True, help="Path to write JSON result")
    args = parser.parse_args()

    spec = json.loads(open(args.spec).read())

    X_train = np.load(spec["X_train_path"])
    y_train = np.load(spec["y_train_path"])
    X_val = np.load(spec["X_val_path"])
    y_val = np.load(spec["y_val_path"])

    model = train_gbm(X_train, y_train, n_classes=spec["n_classes"], seed=spec.get("seed", 42))
    metrics = evaluate_gbm(model, X_val, y_val)

    with open(args.out, "w") as f:
        json.dump(metrics, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
