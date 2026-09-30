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


