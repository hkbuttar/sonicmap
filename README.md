# SonicMap — Genre/Mood Classification & Audio-Native Similarity Search
Music genre and valence-arousal mood modeling with audio-native similarity search. CNN and gradient-boosted classifiers, a contrastive similarity embedding compared against a classification-derived one, cross-dataset validation, and playlist generation. Real audio, CPU-only.

## Step 5: mood regression

Run the DEAM valence-arousal comparison after downloading DEAM and caching its features:

```bash
python -m features.build_features --audio-root data/raw/mood/MEMD_audio --cache-dir data/cache/features/deam
python -m mood.run_experiment --n-folds 5 --epochs 15
```

The experiment predicts DEAM's real song-level continuous valence and arousal annotations. It reports MAE, RMSE, and R² separately for each coordinate, with fold-level 95% t confidence intervals, for both a mel-spectrogram CNN and gradient-boosted engineered-feature regressors.
