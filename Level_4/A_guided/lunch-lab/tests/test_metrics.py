import json

import pytest

from lunchlab.metrics import (
    accuracy,
    classification_report,
    confusion_matrix,
    f1_score,
    precision,
    recall,
)

Y_TRUE = [1, 1, 0, 0, 1, 0]
Y_PRED = [1, 0, 0, 1, 1, 0]
# TN = 2, FP = 1, FN = 1, TP = 2


def test_confusion_matrix():
    assert confusion_matrix(Y_TRUE, Y_PRED).tolist() == [[2, 1], [1, 2]]


def test_precision_recall_f1():
    assert precision(Y_TRUE, Y_PRED) == pytest.approx(2 / 3)
    assert recall(Y_TRUE, Y_PRED) == pytest.approx(2 / 3)
    assert f1_score(Y_TRUE, Y_PRED) == pytest.approx(2 / 3)


def test_accuracy():
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(4 / 6)
    assert accuracy([1, 0], [1, 0]) == pytest.approx(1.0)
    assert accuracy([1, 0], [0, 1]) == pytest.approx(0.0)


def test_accuracy_rejects_empty_and_mismatched():
    with pytest.raises(ValueError):
        accuracy([], [])
    with pytest.raises(ValueError):
        accuracy([1, 0], [1])


def test_classification_report():
    report = classification_report(Y_TRUE, Y_PRED)
    assert report["accuracy"] == pytest.approx(4 / 6)
    assert report["precision"] == pytest.approx(2 / 3)
    assert report["recall"] == pytest.approx(2 / 3)
    assert report["f1"] == pytest.approx(2 / 3)
    assert report["confusion_matrix"] == [[2, 1], [1, 2]]



