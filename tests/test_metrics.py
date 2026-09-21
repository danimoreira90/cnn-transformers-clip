"""Unit tests for classification metrics.

The hand-built confusion matrix below is deliberately awkward: class 2 exists in the
data but is never predicted. That is the shape of the failure activity 4.1 asks us to
diagnose, so it is the default fixture rather than an edge case bolted on at the end.

    y_true = [0,0,0,0, 1,1,1, 2,2]
    y_pred = [0,0,0,1, 1,1,0, 0,1]

    confusion (rows = true, columns = predicted)
                pred 0  pred 1  pred 2
        true 0       3       1       0
        true 1       1       2       0
        true 2       1       1       0
"""

from __future__ import annotations

import numpy as np
import pytest

from cv_mba.metrics import (
    ZeroSupportError,
    accuracy,
    confusion_matrix,
    macro_f1,
    per_class_accuracy,
    per_class_precision,
    per_class_recall,
)

Y_TRUE = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])
Y_PRED = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1])
CM = np.array([[3, 1, 0], [1, 2, 0], [1, 1, 0]])


def test_confusion_matrix_matches_the_hand_built_table() -> None:
    assert np.array_equal(confusion_matrix(Y_TRUE, Y_PRED, n_classes=3), CM)


def test_confusion_matrix_rows_are_true_labels_not_predictions() -> None:
    # One true 0 predicted as 1, and nothing the other way round. If the orientation
    # were flipped this would land at [1][0] instead of [0][1].
    cm = confusion_matrix(np.array([0]), np.array([1]), n_classes=2)

    assert cm[0][1] == 1
    assert cm[1][0] == 0


def test_confusion_matrix_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        confusion_matrix(np.array([0, 1]), np.array([0]), n_classes=2)


def test_confusion_matrix_rejects_a_label_outside_the_class_range() -> None:
    with pytest.raises(ValueError, match="outside"):
        confusion_matrix(np.array([0, 2]), np.array([0, 0]), n_classes=2)


def test_confusion_matrix_rejects_fewer_than_two_classes() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        confusion_matrix(np.array([0]), np.array([0]), n_classes=1)


def test_per_class_recall_matches_hand_computation() -> None:
    recall = per_class_recall(CM)

    assert recall == pytest.approx([3 / 4, 2 / 3, 0.0])


def test_per_class_recall_raises_when_a_class_has_no_true_examples() -> None:
    cm = np.array([[3, 1, 0], [1, 2, 0], [0, 0, 0]])

    with pytest.raises(ZeroSupportError, match="2"):
        per_class_recall(cm)


def test_per_class_accuracy_is_recall_and_says_so() -> None:
    assert per_class_accuracy(CM) == pytest.approx(per_class_recall(CM))
    assert "recall" in (per_class_accuracy.__doc__ or "")


def test_per_class_precision_matches_hand_computation() -> None:
    precision = per_class_precision(CM)

    assert precision[0] == pytest.approx(3 / 5)
    assert precision[1] == pytest.approx(2 / 4)


def test_per_class_precision_is_undefined_for_a_class_never_predicted() -> None:
    precision = per_class_precision(CM)

    assert np.isnan(precision[2])


def test_macro_f1_is_zero_not_nan_for_a_class_never_predicted() -> None:
    # Class 2 has undefined precision but zero recall, so it scored nothing. Zero is the
    # honest value; a nan here would silently poison the macro average.
    expected = (2 * 0.6 * 0.75 / 1.35 + 2 * 0.5 * (2 / 3) / (0.5 + 2 / 3) + 0.0) / 3

    assert macro_f1(CM) == pytest.approx(expected)
    assert not np.isnan(macro_f1(CM))


def test_accuracy_counts_the_diagonal_over_everything() -> None:
    assert accuracy(CM) == pytest.approx(5 / 9)


def test_perfect_predictions_score_one_everywhere() -> None:
    cm = np.array([[4, 0, 0], [0, 3, 0], [0, 0, 2]])

    assert per_class_recall(cm) == pytest.approx([1.0, 1.0, 1.0])
    assert per_class_precision(cm) == pytest.approx([1.0, 1.0, 1.0])
    assert macro_f1(cm) == pytest.approx(1.0)
    assert accuracy(cm) == pytest.approx(1.0)


def test_high_global_accuracy_can_hide_a_minority_class_scoring_zero() -> None:
    # The failed X-ray project in activity 4.1: 840 Normal, 240 Pneumonia, 120 COVID,
    # and a model that answers Normal every time. Global accuracy looks like a pass.
    # COVID recall is the number the clinical team was reacting to.
    cm = np.array([[840, 0, 0], [240, 0, 0], [120, 0, 0]])

    assert accuracy(cm) == pytest.approx(0.70)
    assert per_class_recall(cm) == pytest.approx([1.0, 0.0, 0.0])
    assert macro_f1(cm) == pytest.approx((2 * 0.7 * 1.0 / 1.7) / 3)
