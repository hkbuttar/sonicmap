# Results and Honest Comparison

## Headline results

| Question | Result |
|---|---|
| Did augmentation help? | CNN accuracy changed by +0.085 (95% CI -0.003 to +0.173); GBM changed by +0.014 (-0.017 to +0.045). |
| Best genre classifier | Augmented CNN: 0.759 accuracy versus augmented GBM: 0.732. |
| Mood regression | GBM reached R²=0.395 valence and R²=0.446 arousal; gains over CNN were small. |
| Best in-sample audio retrieval | Classification embedding P@10=0.890; triplet embedding P@10=0.677. |
| Cross-dataset classification | FMA accuracy=0.346, an absolute drop of 0.413 from GTZAN CV. |
| More robust embedding OOD | Chance-adjusted P@10 drop: classification=0.419, triplet=0.221. The triplet space degraded less but remained slightly weaker on FMA. |
| Playlist coherence lift over random | Classification=+0.390; triplet=+0.243. |

## Honest interpretation

Augmentation produced a substantial observed CNN gain at this dataset size, but the five-fold paired 95% interval narrowly included zero, so the evidence is promising rather than conclusive. Its GBM effect was small and uncertain. Mood remained substantially harder and noisier than genre classification; GBM only narrowly led the CNN.

The purpose-built triplet objective reduced its training loss, but it did **not** beat the simpler classification-derived embedding on genre-based retrieval, silhouette score, or playlist lift. Its positive result was robustness: after correcting for the three-class FMA versus ten-class GTZAN evaluation, its cross-dataset retrieval drop was smaller.

The genre CNN generalized poorly to FMA, confirming a large distribution gap. Similarity evaluation retrieval is in-sample and uses genre agreement as a weak proxy, so its high scores should not be interpreted as human perceptual similarity. The metadata baseline is an oracle-like ceiling because it directly uses the evaluation label. Playlist comparisons are valid as within-space lifts over random; raw cosine values across separately trained spaces are not directly comparable.

## Scope and limitations

- GTZAN is small and contains known duplicates, label faults, and artifacts.
- FMA evaluation used 2,999 tracks from only the three defensible exact-overlap genres.
- DEAM annotations are subjective and include annotator disagreement.
- Similarity and playlist quality still require the generated human-review sheets.
