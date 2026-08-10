"""Build and execute research.ipynb using the standard library.

This keeps the checked-in notebook reproducible even in the CPU project
environment, where Jupyter/nbformat are intentionally not runtime dependencies.
"""

import ast
import base64
import contextlib
import io
import json
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent / "research.ipynb"


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


CELLS = [
    markdown("""# SonicMap Research Report

Genre and continuous valence–arousal modeling, audio-native similarity search, cross-dataset validation, and playlist generation on real audio.

This notebook is an executed view of the canonical CSV/JSON experiment artifacts from Steps 4–11. It does not retrain models, so it remains quick to reproduce on CPU. All values below are loaded from disk rather than manually transcribed.
"""),
    markdown("""## Experimental scope

- **Genre:** GTZAN, 999 successfully processed tracks across 10 genres, with known duplicate and labeling limitations.
- **Mood:** DEAM, 1,802 excerpts with real continuous valence and arousal annotations.
- **Out-of-distribution evaluation:** 2,999 FMA Small tracks from the only three exact taxonomy overlaps: Hip-Hop, Pop, and Rock. One corrupt FMA MP3 was skipped.
- **Uncertainty:** five-fold t confidence intervals for model experiments and query-level bootstrap intervals for retrieval and playlists.
"""),
    code("""from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)
print(f"Artifact root: {ROOT}")
"""),
    markdown("""## 1. Genre classification and augmentation

The CNN and gradient-boosted model were evaluated with stratified five-fold cross-validation. Augmented variants were confined to the training side of each fold to prevent near-duplicate leakage.
"""),
    code("""genre = pd.read_csv(ROOT / "classification/results/step4_classification_summary.csv")
accuracy = genre[genre.metric == "accuracy"].copy()
accuracy["condition"] = accuracy["model"].str.upper() + np.where(accuracy["augmented"], " + augmentation", "")

fig, ax = plt.subplots(figsize=(8, 4))
yerr = np.vstack([accuracy["mean"] - accuracy["ci_low"], accuracy["ci_high"] - accuracy["mean"]])
ax.bar(accuracy["condition"], accuracy["mean"], yerr=yerr, capsize=5, color=["#729ECE", "#4C78A8", "#F2CF5B", "#E6A700"])
ax.set(ylim=(0.5, 0.85), ylabel="Accuracy", title="GTZAN genre classification (mean and 95% CI)")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
accuracy[["model", "augmented", "mean", "ci_low", "ci_high"]]
"""),
    markdown("""The augmented CNN had the best observed accuracy (0.759). Its paired gain was 0.085, although the five-fold 95% interval narrowly included zero; this is promising rather than conclusive. The GBM augmentation effect was much smaller.
"""),
    markdown("""## 2. Continuous mood regression

Valence and arousal were modeled as continuous DEAM targets. R² is shown because it expresses improvement over a mean-target predictor; MAE and RMSE remain in the source artifact.
"""),
    code("""mood = pd.read_csv(ROOT / "mood/results/step5_mood_regression_summary.csv")
r2 = mood[mood.metric.str.endswith("_r2")].copy()
r2["target"] = r2.metric.str.replace("_r2", "", regex=False)

fig, ax = plt.subplots(figsize=(7, 4))
for offset, (model, group) in zip((-0.18, 0.18), r2.groupby("model")):
    x = np.arange(len(group)) + offset
    ax.bar(x, group["mean"], width=.36, label=model.upper())
    ax.errorbar(x, group["mean"], yerr=[group["mean"]-group["ci_low"], group["ci_high"]-group["mean"]], fmt="none", color="black", capsize=4)
ax.set_xticks(range(2), ["Valence", "Arousal"])
ax.set(ylabel="R²", title="DEAM mood regression (mean and 95% CI)")
ax.legend(frameon=False)
fig.tight_layout()
r2[["model", "target", "mean", "ci_low", "ci_high"]]
"""),
    markdown("""Mood was harder and noisier than genre classification, as expected for subjective annotations. GBM narrowly led the CNN, reaching R² 0.395 for valence and 0.446 for arousal.
"""),
    markdown("""## 3. Learned embedding spaces

The classification embedding reuses the genre CNN's 128-dimensional penultimate layer. The triplet network was trained separately with explicit same-genre positives and cross-genre negatives.
"""),
    code("""projection_files = {
    "Classification-derived": ROOT / "embeddings/results/classification/classification_projection.csv",
    "Triplet-loss": ROOT / "embeddings/results/triplet/triplet_projection.csv",
}
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (title, path) in zip(axes, projection_files.items()):
    frame = pd.read_csv(path)
    for label, group in frame.groupby("label"):
        ax.scatter(group.x, group.y, s=9, alpha=.65, label=label)
    ax.set(title=title, xlabel="UMAP 1", ylabel="UMAP 2")
axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
fig.tight_layout()

embedding_metrics = pd.DataFrame([
    {"embedding": name, **json.loads((ROOT / path).read_text())}
    for name, path in {
        "classification": "embeddings/results/classification/classification_embedding_metrics.json",
        "triplet": "embeddings/results/triplet/triplet_embedding_metrics.json",
    }.items()
])
embedding_metrics[["embedding", "embedding_silhouette_cosine", "projection_silhouette_euclidean"]]
"""),
    markdown("""The purpose-built objective learned—the triplet loss fell from 0.112 to 0.059—but it did not surpass the simpler classification representation on genre silhouette or downstream in-distribution retrieval.
"""),
    markdown("""## 4. Four-way similarity search

Precision@k uses same-genre agreement as a weak automated relevance signal. The metadata method directly uses that label, making it an oracle-like ceiling rather than audio retrieval.
"""),
    code("""retrieval = pd.read_csv(ROOT / "similarity/results/step8_similarity_comparison.csv")
fig, ax = plt.subplots(figsize=(8, 5))
for method, group in retrieval.groupby("method", sort=False):
    ax.plot(group.k, group.precision_at_k, marker="o", label=method.replace("_", " "))
    ax.fill_between(group.k, group.ci_low, group.ci_high, alpha=.12)
ax.set(xlabel="k", ylabel="Precision@k", ylim=(0.35, 1.03), title="GTZAN similarity retrieval")
ax.legend(frameon=False)
fig.tight_layout()
retrieval[retrieval.k == 10][["method", "precision_at_k", "ci_low", "ci_high"]]
"""),
    markdown("""At k=10, the classification embedding scored 0.890, the triplet embedding 0.677, and engineered features 0.523. These are in-sample GTZAN results and should not be read as human perceptual-similarity scores.
"""),
    markdown("""## 5. Cross-dataset generalization

GTZAN-trained models were evaluated on FMA without retraining. Retrieval drops were chance-adjusted because FMA's exact-overlap evaluation has three classes while GTZAN has ten.
"""),
    code("""generalization = pd.read_csv(ROOT / "generalization/results/step9_classification_generalization.csv")
per_genre = pd.read_csv(ROOT / "generalization/results/step9_per_genre_accuracy.csv")
ood = pd.read_csv(ROOT / "generalization/results/step9_similarity_generalization.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(["GTZAN CV", "FMA OOD"], [generalization.gtzan_cv_accuracy.iloc[0], generalization.accuracy.iloc[0]], color=["#4C78A8", "#E45756"])
axes[0].set(ylim=(0, 0.85), ylabel="Accuracy", title="Genre-classifier distribution shift")
k10 = ood[ood.k == 10]
axes[1].bar(k10.method.str.replace("_embedding", "", regex=False), k10.adjusted_precision_drop, color=["#4C78A8", "#F58518"])
axes[1].set(ylabel="Chance-adjusted P@10 drop", title="Embedding degradation on FMA")
fig.tight_layout()
per_genre[["genre", "n", "accuracy"]]
"""),
    markdown("""Classifier accuracy fell from 0.759 to 0.346, with especially weak FMA Pop accuracy (0.147). The classification embedding retained slightly higher absolute FMA P@10, but the triplet embedding had the smaller chance-adjusted degradation (0.221 versus 0.419), its clearest positive result.
"""),
    markdown("""## 6. Playlist generation

Playlists used a greedy local walk with a gradually increasing target distance from the seed. Each method was compared with random playlists inside its own space; raw cosine levels are not comparable across independently trained spaces.
"""),
    code("""playlists = pd.read_csv(ROOT / "similarity/results/playlists/step10_playlist_coherence.csv")
pairwise = playlists[playlists.metric == "pairwise_similarity"].copy()
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(pairwise.method.str.replace("_embedding", "", regex=False), pairwise.mean_difference,
       yerr=[pairwise.mean_difference-pairwise.difference_ci_low, pairwise.difference_ci_high-pairwise.mean_difference], capsize=5)
ax.set(ylabel="Pairwise-similarity lift over random", title="Playlist coherence (100 seeds)")
fig.tight_layout()
playlists[playlists.metric.isin(["pairwise_similarity", "seed_genre_fraction", "monotonic_drift_fraction"])][
    ["method", "metric", "playlist_mean", "random_mean", "mean_difference", "difference_ci_low", "difference_ci_high"]
]
"""),
    markdown("""Both methods produced coherent, progressively drifting playlists. The classification representation had the larger pairwise-similarity lift over random (0.390 versus 0.243) and greater seed-genre retention.
"""),
    markdown("""## 7. Consolidated findings and limitations
"""),
    code("""comparison = pd.read_csv(ROOT / "results/step11_comparison.csv")
print((ROOT / "results/step11_findings.md").read_text())
comparison
"""),
    markdown("""## 8. Complete findings ledger

The tables below intentionally preserve every reported quantitative result, not only the headline metrics plotted above. They are printed without notebook display truncation and retain confidence intervals, sample counts, and methodological qualifiers from the canonical artifacts.
"""),
    markdown("""### 8.1 Genre classification: every model, condition, and metric
"""),
    code("""genre_all = pd.read_csv(ROOT / "classification/results/step4_classification_summary.csv")
print(genre_all.to_string(index=False))
"""),
    markdown("""### 8.2 Mood regression: MAE, RMSE, and R² for both targets and models
"""),
    code("""mood_all = pd.read_csv(ROOT / "mood/results/step5_mood_regression_summary.csv")
print(mood_all.to_string(index=False))
"""),
    markdown("""### 8.3 Triplet optimization diagnostics

Loss was not strictly monotonic at every epoch, but the overall decline demonstrates that the training objective was optimized. That alone did not guarantee better downstream retrieval.
"""),
    code("""triplet_history = pd.read_csv(ROOT / "embeddings/results/triplet/triplet_training_history.csv")
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(triplet_history.epoch, triplet_history.triplet_loss, marker="o")
ax.set(xlabel="Epoch", ylabel="Triplet loss", title="Triplet training diagnostic")
ax.grid(alpha=.2)
fig.tight_layout()
print(triplet_history.to_string(index=False))
"""),
    markdown("""### 8.4 Similarity search: cosine and Euclidean results

Cosine and Euclidean rankings were effectively identical for the L2-normalized learned embeddings. Engineered features performed slightly better with cosine. The full distance table follows.
"""),
    code("""distance_all = pd.read_csv(ROOT / "similarity/results/step8_distance_comparison.csv")
print(distance_all.to_string(index=False))
"""),
    markdown("""### 8.5 FMA generalization: classifier, per-genre, and retrieval results

The classifier suffered a large absolute distribution-shift loss. Pop was the weakest aligned FMA genre. The classification embedding remained slightly stronger in absolute FMA precision, while the triplet embedding showed the smaller chance-adjusted drop at every reported k.
"""),
    code("""print("CLASSIFICATION")
print(pd.read_csv(ROOT / "generalization/results/step9_classification_generalization.csv").to_string(index=False))
print("\\nPER GENRE")
print(pd.read_csv(ROOT / "generalization/results/step9_per_genre_accuracy.csv").to_string(index=False))
print("\\nSIMILARITY")
print(pd.read_csv(ROOT / "generalization/results/step9_similarity_generalization.csv").to_string(index=False))
"""),
    markdown("""### 8.6 Playlist evaluation: every coherence and drift metric

Both learned spaces beat their own random baselines for pairwise similarity, adjacent similarity, seed-genre retention, and monotonic drift. Generated playlists ended closer to their seeds than random playlists, which is expected from a coherence-controlled traversal. Cross-space conclusions use lift over random, not raw cosine values.
"""),
    code("""playlist_all = pd.read_csv(ROOT / "similarity/results/playlists/step10_playlist_coherence.csv")
print(playlist_all.to_string(index=False))
"""),
    markdown("""### 8.7 Findings not yet resolved automatically

- Genre precision is only a weak proxy for perceptual similarity.
- The metadata baseline is an oracle-like ceiling, not an audio system.
- The generated Step 8 and Step 10 qualitative-review CSVs remain intentionally unrated; no human-listening conclusion is fabricated here.
- Step 8 is in-sample retrieval. Step 9 is the genuine unseen-distribution test.
- FMA used 2,999 of 3,000 selected tracks because one source MP3 was corrupt.
- The triplet embedding's lower cross-dataset degradation is a robustness result, not evidence that it was the best absolute retriever.
"""),
    markdown("""## 9. Testing and validation audit

Step 12 combined the 39-test unit/integration suite with an artifact-backed audit. It checked random baselines, fold integrity, augmentation leakage, triplet optimization and initial-to-final silhouette, FMA alignment, retrieval sanity, and playlist lift.
"""),
    code("""validation_checks = pd.read_csv(ROOT / "validation/results/step12_validation_checks.csv")
print((ROOT / "validation/results/step12_validation_report.md").read_text())
validation_checks
"""),
    markdown("""All 18 critical and positive sanity checks passed. The FMA classifier comparison with a balanced three-class random baseline remained **inconclusive**, because its 95% confidence interval crossed chance. Human perceptual similarity remains unvalidated until the qualitative listening sheets are completed.
"""),
    markdown("""### Reproducibility

Regenerate this executed notebook from the repository root with:

```bash
.venv/bin/python notebooks/build_research_notebook.py
```

Model training is intentionally separate. The notebook consumes saved experiment artifacts, making reporting deterministic and inexpensive while keeping the original training commands auditable in the README.
"""),
]


def execute_cell(cell, namespace, execution_count):
    source = "".join(cell["source"])
    tree = ast.parse(source)
    outputs = []
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            result = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                prefix = ast.Module(body=tree.body[:-1], type_ignores=[])
                exec(compile(prefix, "<research.ipynb>", "exec"), namespace)
                result = eval(compile(ast.Expression(tree.body[-1].value), "<research.ipynb>", "eval"), namespace)
            else:
                exec(compile(tree, "<research.ipynb>", "exec"), namespace)
        if stdout.getvalue():
            outputs.append({"name": "stdout", "output_type": "stream", "text": stdout.getvalue().splitlines(keepends=True)})
        if result is not None:
            outputs.append({
                "data": {"text/plain": repr(result).splitlines(keepends=True)},
                "execution_count": execution_count, "metadata": {}, "output_type": "execute_result",
            })
        plt = namespace.get("plt")
        if plt is not None:
            for figure_number in plt.get_fignums():
                buffer = io.BytesIO()
                plt.figure(figure_number).savefig(buffer, format="png", dpi=120, bbox_inches="tight")
                outputs.append({
                    "data": {"image/png": base64.b64encode(buffer.getvalue()).decode("ascii")},
                    "metadata": {}, "output_type": "display_data",
                })
            plt.close("all")
    except Exception as error:
        outputs.append({
            "ename": type(error).__name__, "evalue": str(error), "output_type": "error",
            "traceback": traceback.format_exc().splitlines(),
        })
        cell["outputs"] = outputs
        raise
    cell["execution_count"] = execution_count
    cell["outputs"] = outputs


def main():
    namespace = {"__name__": "__main__"}
    execution_count = 0
    original_cwd = Path.cwd()
    try:
        import os
        os.chdir(ROOT)
        for cell in CELLS:
            if cell["cell_type"] == "code":
                execution_count += 1
                execute_cell(cell, namespace, execution_count)
    finally:
        os.chdir(original_cwd)
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Wrote executed notebook: {OUTPUT}")


if __name__ == "__main__":
    main()
