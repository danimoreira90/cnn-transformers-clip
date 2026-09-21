"""Differential check: our metrics against scikit-learn on four realistic label sets.

This lives in `evals/` rather than `tests/` on purpose. The unit tests in
`tests/test_metrics.py` specify behaviour with hand-computed numbers, which is the
contract. This file checks those hand-computed numbers against an independent
implementation, which is validation. Keeping them apart means the unit suite never
inherits a dependency on another library's future choices.

sklearn needs to be told to be honest to match us: `zero_division=np.nan` for precision,
because a class that is never predicted has undefined precision, and `zero_division=0`
for macro F1, because a class that found nothing earned nothing. Its defaults report 0.0
for undefined precision, which is the flattering answer this project is built to avoid.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from cv_mba.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    per_class_precision,
    per_class_recall,
)

CASES = {
    "class never predicted": (
        np.array([0, 0, 0, 0, 1, 1, 1, 2, 2]),
        np.array([0, 0, 0, 1, 1, 1, 0, 0, 1]),
        3,
    ),
    "perfect": (np.array([0, 1, 2, 0, 1, 2]), np.array([0, 1, 2, 0, 1, 2]), 3),
    "xray baseline answers Normal every time": (
        np.repeat([0, 1, 2], [840, 240, 120]),
        np.zeros(1200, dtype=int),
        3,
    ),
    "elpv binary, imbalanced 1803/821": (
        np.repeat([0, 1], [1803, 821]),
        np.concatenate(
            [
                np.zeros(1700, dtype=int),
                np.ones(103, dtype=int),
                np.ones(600, dtype=int),
                np.zeros(221, dtype=int),
            ]
        ),
        2,
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_metrics_agree_with_sklearn(name: str) -> None:
    y_true, y_pred, n_classes = CASES[name]
    labels = range(n_classes)
    cm = confusion_matrix(y_true, y_pred, n_classes)

    assert np.array_equal(cm, sk_confusion_matrix(y_true, y_pred, labels=labels))
    assert np.allclose(
        per_class_recall(cm),
        recall_score(y_true, y_pred, average=None, labels=labels, zero_division=np.nan),
        equal_nan=True,
    )
    assert np.allclose(
        per_class_precision(cm),
        precision_score(
            y_true, y_pred, average=None, labels=labels, zero_division=np.nan
        ),
        equal_nan=True,
    )
    assert macro_f1(cm) == pytest.approx(
        f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    )
    assert accuracy(cm) == pytest.approx(accuracy_score(y_true, y_pred))
