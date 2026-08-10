# Datasets

All raw audio and annotation files live under `data/raw/` and are
git-ignored — fetch them with the scripts in `scripts/download_*.py`.

## GTZAN (primary genre dataset)

- 1,000 tracks, 30s each, 10 genres (100 tracks/genre): blues, classical,
  country, disco, hiphop, jazz, metal, pop, reggae, rock.
- Fetched via `scripts/download_gtzan.py` from a Hugging Face mirror
  (`marsyas/gtzan`) — the original host (opihi.cs.uvic.ca /
  marsyas.info) has been unreachable for years.
- **Known issues**, per Sturm (2013), *"The GTZAN dataset: Its contents,
  its faults, their effects on evaluation, and its future use"*:
  - Duplicate tracks (~50 exact or near-exact repeats across genre folders).
  - Mislabeled tracks (some tracks placed in a genre folder that
    disagrees with expert re-labeling).
  - Distortion artifacts on some clips.
  - Skewed toward artists whose work spans multiple genre labels.
  - This project uses GTZAN as-is rather than hand-filtering it — the
    faults are disclosed here and referenced in the README/results
    rather than silently corrected, since "cleaning" GTZAN without a
    principled, published relabeling introduces its own bias.

## FMA subset (secondary, cross-dataset generalization only)

- `fma_small`: 8,000 tracks, 30s clips, 8 balanced genres, ~7.2GB.
- `fma_metadata`: track/genre/artist tables (`tracks.csv`, `genres.csv`,
  etc.), ~342MB.
- Fetched via `scripts/download_fma.py` from the official FMA mirror
  (`os.unil.cloud.switch.ch/fma`). Source: https://github.com/mdeff/fma
- Deliberately the *smallest* official FMA split (not `fma_medium`,
  `fma_large`, or the full 917GB `fma_full` corpus) — this project uses
  it only as an out-of-distribution eval set for the genre classifier and
  both embeddings, not for training, so a targeted pull is the
  right scope.
- Genre taxonomy differs from GTZAN's (FMA's top-level genres are
  broader/differently defined), so cross-dataset genre alignment uses a
  manual label mapping — documented where implemented in
  `generalization/`, not assumed to be 1:1.

## DEAM (valence-arousal regression targets)

- MediaEval Database for Emotional Analysis in Music: 1,802 excerpts (45s
  each) with continuous valence/arousal annotations from multiple
  annotators (per Russell's circumplex model).
- Fetched via `scripts/download_deam.py` from the official host
  (cvml.unige.ch/databases/DEAM). ~1.3GB audio + ~5MB annotations.
- Used directly as regression targets rather than deriving a mood proxy
  from genre labels — this is real, if noisy and inherently subjective,
  ground truth. Annotator disagreement is itself a documented property of
  DEAM and is expected to cap achievable regression accuracy regardless
  of model choice.

## Why not the other options

- **PMEmo** was considered as an alternative mood dataset; DEAM was
  chosen instead for its larger annotator pool and status as the
  standard MediaEval benchmark, making comparisons to published
  valence-arousal results more meaningful.
- **Full FMA** (`fma_large`/`fma_full`) was rejected for this project's
  scope: it exists only to test generalization, not to train on, so
  its ~13x–115x larger size over `fma_small` buys nothing here.
