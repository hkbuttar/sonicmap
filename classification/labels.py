"""Fixed genre label ordering, shared across CNN and GBM so class indices
line up between the two approaches and across folds."""

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
LABEL_TO_IDX = {g: i for i, g in enumerate(GENRES)}
IDX_TO_LABEL = {i: g for g, i in LABEL_TO_IDX.items()}
