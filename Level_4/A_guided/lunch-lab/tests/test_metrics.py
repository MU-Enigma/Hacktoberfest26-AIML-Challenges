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


def test_precision_is_zero_when_nothing_predicted_positive():
    # TP + FP = 0, so the denominator is zero
    assert precision([1, 0, 1], [0, 0, 0]) == 0.0


def test_recall_is_zero_when_there_are_no_true_positives_to_find():
    # TP + FN = 0, so the denominator is zero
    assert recall([0, 0, 0], [1, 0, 1]) == 0.0


def test_f1_is_zero_when_precision_and_recall_are_both_zero():
    # No denominator is zero for precision or recall here, but both are 0, so p + r = 0
    assert f1_score([1, 0], [0, 1]) == 0.0
    # Nothing predicted positive and nothing to find: every denominator is zero
    assert f1_score([0, 0, 0], [0, 0, 0]) == 0.0


