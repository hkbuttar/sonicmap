# SonicMap — Genre/Mood Classification & Audio-Native Similarity Search
Music genre and valence-arousal mood modeling with audio-native similarity search. CNN and gradient-boosted classifiers, a contrastive similarity embedding compared against a classification-derived one, cross-dataset validation, and playlist generation. Real audio, CPU-only.

## Step 5: mood regression

Run the DEAM valence-arousal comparison after downloading DEAM and caching its features:

```bash
python -m features.build_features --audio-root data/raw/mood/MEMD_audio --cache-dir data/cache/features/deam
python -m mood.run_experiment --n-folds 5 --epochs 15
```

The experiment predicts DEAM's real song-level continuous valence and arousal annotations. It reports MAE, RMSE, and R² separately for each coordinate, with fold-level 95% t confidence intervals, for both a mel-spectrogram CNN and gradient-boosted engineered-feature regressors.

## Step 6: classification-derived embedding

Train the best Step 4 CNN condition on all GTZAN training data, extract its 128-dimensional penultimate layer for each original track, and create a UMAP projection:

```bash
.venv/bin/python -m embeddings.run_classification_embedding --epochs 15
```

Artifacts are written to `embeddings/results/classification/`: L2-normalized embeddings for similarity search, track metadata and 2D coordinates, a projection plot, cosine silhouette score against genre labels, and the full-data CNN checkpoint. Use `--projection tsne` for t-SNE or `--no-augmentation` to train only on original tracks.

## Step 7: triplet-loss embedding

Train a separate similarity network with explicit genre-supervised anchor/positive/negative triplets:

```bash
.venv/bin/python -m embeddings.run_triplet_embedding --epochs 15
```

Same-genre positives are sampled from a different source track, preventing an original clip and its augmentation from forming a trivial pair; negatives come from another genre. The default uses 2,000 triplets per epoch and margin `0.2`. Artifacts in `embeddings/results/triplet/` use the same ordering and 128-dimensional normalized format as Step 6, and include loss history, UMAP coordinates, silhouette scores, a checkpoint, and the silhouette delta versus the classification-derived embedding.

## Step 8: similarity search and evaluation

Evaluate exact nearest-neighbor retrieval in both learned spaces and the engineered-feature and metadata baselines:

```bash
.venv/bin/python -m similarity.run_evaluation
```

The primary four-way table reports genre-label precision@1/5/10/20 with query-level bootstrap confidence intervals. A detailed table additionally compares cosine and Euclidean distance, while `step8_neighbors.parquet` stores ranked results. The generated qualitative-review CSV is intentionally left for human ratings; genre agreement is disclosed as a weak automated proxy, and the genre-metadata baseline as an oracle-like upper bound rather than audio-native retrieval.

## Step 9: cross-dataset generalization

Cache the exact-overlap portion of FMA Small, then evaluate the GTZAN-trained classifier and both embeddings without retraining:

```bash
.venv/bin/python -m generalization.build_fma_features
.venv/bin/python -m generalization.run_evaluation
```

The conservative taxonomy alignment is `Hip-Hop → hiphop`, `Pop → pop`, and `Rock → rock` (3,000 FMA tracks). The five other FMA Small genres are excluded because they lack exact GTZAN equivalents. Reports include classification accuracy/F1, per-genre accuracy, retrieval precision@k, bootstrap intervals, and chance-adjusted retrieval drops to account for FMA's three-class evaluation versus GTZAN's ten classes.

## Step 10: playlist generation

Generate gradually drifting playlists in both learned embedding spaces and compare their coherence with matched random playlists:

```bash
.venv/bin/python -m similarity.run_playlist_evaluation
```

The default evaluates 100 stratified seeds with 50 random playlists per seed and creates one demo playlist per genre for each embedding. Reports include average pairwise and adjacent cosine similarity, seed-genre retention, drift behavior, bootstrap confidence intervals, generated track sequences, and a blank qualitative-review sheet. Pass `--seed-track blues/blues.00000` to generate demos from a specific track.

## Step 11: consolidated results

Regenerate the full comparison table and honest findings report from all completed experiment artifacts:

```bash
.venv/bin/python -m results.compile_results
```

The generated `results/step11_comparison.csv` preserves metrics, confidence intervals, sources, and caveats; `results/step11_findings.md` summarizes augmentation, genre and mood modeling, embedding quality, cross-dataset degradation, and playlist coherence without treating weak genre proxies or oracle metadata as perceptual ground truth.
