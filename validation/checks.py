"""Reusable validation checks with explicit severity and evidence."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Check:
    name: str
    status: str
    observed: float | str
    threshold: float | str
    severity: str
    evidence: str

    def to_dict(self):
        return asdict(self)


def greater_than_check(name, observed, threshold, evidence, severity="critical"):
    return Check(
        name=name, status="pass" if observed > threshold else "fail",
        observed=float(observed), threshold=f"> {threshold}", severity=severity,
        evidence=evidence,
    )


def interval_vs_baseline_check(name, mean, low, high, baseline, evidence, severity="informational"):
    """Classify a confidence interval without treating uncertainty as failure."""
    if low > baseline:
        status = "pass"
    elif high < baseline:
        status = "fail"
    else:
        status = "inconclusive"
    return Check(
        name=name, status=status, observed=f"{mean:.6f} [{low:.6f}, {high:.6f}]",
        threshold=f"CI versus {baseline}", severity=severity, evidence=evidence,
    )


def validate_fold_partition(folds, n_samples):
    validation_counts = np.zeros(n_samples, dtype=int)
    for train_indices, validation_indices in folds:
        if np.intersect1d(train_indices, validation_indices).size:
            return False
        validation_counts[validation_indices] += 1
    return bool(np.all(validation_counts == 1))
