import numpy as np

from classification.cv import make_folds
from validation.checks import interval_vs_baseline_check, validate_fold_partition


def test_stratified_folds_cover_every_sample_once():
    labels = np.repeat(np.arange(3), 10)
    folds = make_folds(labels, n_splits=5, seed=7)
    assert validate_fold_partition(folds, len(labels))


def test_fold_validator_rejects_train_validation_overlap():
    assert not validate_fold_partition([(np.array([0, 1]), np.array([1, 2]))], 3)


def test_interval_check_preserves_inconclusive_result():
    check = interval_vs_baseline_check("example", .34, .32, .36, 1 / 3, "evidence")
    assert check.status == "inconclusive"


def test_interval_check_passes_only_when_entire_interval_beats_baseline():
    assert interval_vs_baseline_check("example", .7, .6, .8, .5, "evidence").status == "pass"
