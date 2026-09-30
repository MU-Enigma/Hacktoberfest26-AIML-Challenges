"""Evaluation metrics for binary classification (labels 0 and 1, positive class is 1)."""

from typing import Any, Dict

import numpy as np


def _check(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Convert to arrays and check they are non-empty and the same length."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    return y_true, y_pred


def _counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Return the four counts (TN, FP, FN, TP) as plain ints."""
    y_true, y_pred = _check(y_true, y_pred)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return the 2x2 confusion matrix [[TN, FP], [FN, TP]].

    Rows are the true label (0 then 1), columns are the predicted label (0 then 1).
    """
    tn, fp, fn, tp = _counts(y_true, y_pred)
    return np.array([[tn, fp], [fn, tp]])


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions that match the true labels."""
    raise NotImplementedError("accuracy is not implemented yet")


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TP / (TP + FP). Returns 0.0 if nothing was predicted positive."""
    tn, fp, fn, tp = _counts(y_true, y_pred)
    return float(tp / (tp + fp))


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TP / (TP + FN). Returns 0.0 if there are no true positives to find."""
    tn, fp, fn, tp = _counts(y_true, y_pred)
    return float(tp / (tp + fn))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Harmonic mean of precision and recall. Returns 0.0 if both are 0."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return float(2 * p * r / (p + r))


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Return accuracy, precision, recall, f1 and the confusion matrix in one dict.

    Keys: "accuracy", "precision", "recall", "f1" (floats) and
    "confusion_matrix" (a list of two lists, [[TN, FP], [FN, TP]]).
    """
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
