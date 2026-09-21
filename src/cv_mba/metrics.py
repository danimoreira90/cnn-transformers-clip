"""Classification metrics, built so that a model ignoring a class cannot look fine.

Global accuracy is the metric the failed X-ray project in activity 4.1 reported, and it
is the reason nobody noticed the model was answering "Normal" to every image. Every
function here is built around that failure:

- A class with no true examples raises. That is a broken split, not a score.
- A class the model never predicts has undefined precision, reported as nan rather than
  as a flattering zero or a quiet one.
- Macro F1 gives a class that scored nothing exactly zero, so the nan cannot leak into
  the average and hide the hole.

Numpy only, no torch. These run on CPU in microseconds and are used by every activity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ZeroSupportError(ValueError):
    """Raised when a class has no true examples, so its recall is not a number.

    This is a data problem, not a modelling result: either the split lost the class or
    the class count is wrong. Returning nan here would let a broken split reach a report.
    """


def confusion_matrix(
    y_true: NDArray[np.integer],
    y_pred: NDArray[np.integer],
    n_classes: int,
) -> NDArray[np.int64]:
    """Count predictions into a table with true labels as rows, predictions as columns.

    Row i, column j holds the number of items whose true class is i and predicted class
    is j. The diagonal is correct predictions.
    """
    if n_classes < 2:
        raise ValueError(f"need at least 2 classes, got {n_classes}")

    true = np.asarray(y_true)
    pred = np.asarray(y_pred)

    if true.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {true.shape} and {pred.shape}"
        )

    for name, values in (("y_true", true), ("y_pred", pred)):
        if values.size and (values.min() < 0 or values.max() >= n_classes):
            raise ValueError(
                f"{name} holds a label outside the range 0..{n_classes - 1}"
            )

    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    return matrix


def per_class_recall(cm: NDArray[np.integer]) -> NDArray[np.float64]:
    """Fraction of each class's true examples that the model found.

    Raises ZeroSupportError naming any class with no true examples.
    """
    support = cm.sum(axis=1)
    empty = np.flatnonzero(support == 0)
    if empty.size:
        names = ", ".join(str(index) for index in empty)
        raise ZeroSupportError(f"no true examples for class(es): {names}")

    return np.diag(cm) / support


def per_class_accuracy(cm: NDArray[np.integer]) -> NDArray[np.float64]:
    """Accuracy within each class, which is that class's recall.

    The marking sheet asks for "accuracy per class". For a single-label problem that is
    recall: of everything truly in this class, how much did the model get right. It is
    not the one-versus-rest figure (TP + TN) / total, which counts every correctly
    rejected outsider and therefore looks excellent for a rare class the model never
    predicts. Reporting that number for the COVID class would reproduce the exact defect
    activity 4.1 asks us to diagnose.
    """
    return per_class_recall(cm)


def per_class_precision(cm: NDArray[np.integer]) -> NDArray[np.float64]:
    """Fraction of each class's predictions that were right.

    A class the model never predicts has no predictions to be right about, so its
    precision is nan rather than 0.0. Zero would read as "predicted badly" when the truth
    is "never predicted at all", and those call for different fixes.
    """
    predicted = cm.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(predicted > 0, np.diag(cm) / predicted, np.nan)

    return precision.astype(np.float64)


def macro_f1(cm: NDArray[np.integer]) -> float:
    """Unweighted mean F1 across classes, so a small class counts as much as a large one.

    A class with zero recall scores zero, whatever its precision does. It found none of
    its examples, so it earned nothing, and that keeps an undefined precision from
    turning the whole average into nan.
    """
    recall = per_class_recall(cm)
    precision = per_class_precision(cm)

    scored = (recall > 0) & np.isfinite(precision)
    f1 = np.zeros_like(recall)
    f1[scored] = (
        2
        * precision[scored]
        * recall[scored]
        / (precision[scored] + recall[scored])
    )

    return float(f1.mean())


def accuracy(cm: NDArray[np.integer]) -> float:
    """Share of all items predicted correctly.

    Reported alongside per-class recall, never alone. On the 840/240/120 X-ray split a
    model that answers Normal every time scores 0.70 here and 0.0 recall on COVID.
    """
    total = cm.sum()
    if total == 0:
        raise ZeroSupportError("confusion matrix is empty")

    return float(np.trace(cm) / total)
