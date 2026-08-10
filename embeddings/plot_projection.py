"""Render an embedding projection in a Torch-free subprocess."""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--title", default="Genre CNN penultimate embeddings")
    args = parser.parse_args()
    frame = pd.read_csv(args.projection)

    figure, axis = plt.subplots(figsize=(10, 8))
    for genre, group in frame.groupby("label", sort=True):
        axis.scatter(group["x"], group["y"], s=14, alpha=0.7, label=genre)
    axis.set(
        title=f"{args.title} — {args.method.upper()}",
        xlabel="Component 1", ylabel="Component 2",
    )
    axis.legend(markerscale=1.5, frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(args.out, dpi=160)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
