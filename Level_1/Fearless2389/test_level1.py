"""
test_level1.py -- pytest tests for similarity.py and squashing.py.

Run with:  python -m pytest -q
from inside this folder (or `python -m pytest Level_1/Fearless2389 -q` from
the repo root -- pytest finds this file either way).
"""

import math

import pytest

from similarity import (
    cosine_similarity,
    dot_product,
    min_max_scale_value,
)
from squashing import relu, sigmoid, sigmoid_naive, tanh


# ---------------------------------------------------------------------------
# dot_product
# ---------------------------------------------------------------------------

def test_dot_product_basic():
    """3*4 + 1*2 = 14 -- a simple, hand-checkable case."""
    assert dot_product([3, 1], [4, 2]) == 14


def test_dot_product_with_zero_vector():
    """Anything dotted with an all-zero vector is 0."""
    assert dot_product([5, -2, 3], [0, 0, 0]) == 0


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical_vectors_is_one():
    """A vector compared with itself points in exactly the same direction."""
    v = [124.0, 210.0, 0.78, 0.82]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    """Perpendicular vectors (e.g. the x-axis and y-axis) have cosine 0."""
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_minus_one():
    """A vector and its exact opposite point in opposite directions."""
    assert cosine_similarity([1, 2], [-1, -2]) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_does_not_crash():
    """A zero-length vector has no direction -- we define this as 0,
    not a ZeroDivisionError.
    """
    assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


# ---------------------------------------------------------------------------
# min_max_scale_value
# ---------------------------------------------------------------------------

def test_min_max_scale_value_endpoints():
    """The min maps to 0.0 and the max maps to 1.0."""
    assert min_max_scale_value(10, 10, 20) == pytest.approx(0.0)
    assert min_max_scale_value(20, 10, 20) == pytest.approx(1.0)


def test_min_max_scale_value_midpoint():
    """A value halfway between min and max scales to 0.5."""
    assert min_max_scale_value(15, 10, 20) == pytest.approx(0.5)


def test_min_max_scale_value_equal_min_max_returns_half():
    """If every song has the same value for a feature, min == max, and we
    return 0.5 instead of dividing by zero.
    """
    assert min_max_scale_value(7, 7, 7) == 0.5


def test_min_max_scale_value_can_scale_a_point_outside_the_original_range():
    """The loved song is scaled using the 10 songs' min/max, and it is
    allowed to land outside [0, 1] if it's more extreme than any of them.
    """
    assert min_max_scale_value(25, 10, 20) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------

def test_sigmoid_of_zero_is_half():
    assert sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_large_positive_input_approaches_one():
    assert sigmoid(50.0) == pytest.approx(1.0, abs=1e-6)


def test_sigmoid_large_negative_input_approaches_zero():
    assert sigmoid(-50.0) == pytest.approx(0.0, abs=1e-6)


def test_sigmoid_handles_extreme_negative_input_without_crashing():
    """This is the exact case that breaks the naive formula: an input far
    enough negative that e^(-x) overflows a float. Our stable sigmoid must
    handle it and return (very close to) 0.0.
    """
    assert sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-10)
    assert sigmoid(-10000.0) == 0.0


def test_sigmoid_handles_extreme_positive_input_without_crashing():
    assert sigmoid(1000.0) == pytest.approx(1.0, abs=1e-10)


def test_naive_sigmoid_overflows_on_large_negative_input():
    """Document the bug: the textbook formula really does raise
    OverflowError once -x is large enough that math.exp(-x) can't be
    represented as a float (around x <= -710).
    """
    with pytest.raises(OverflowError):
        sigmoid_naive(-1000.0)


# ---------------------------------------------------------------------------
# relu
# ---------------------------------------------------------------------------

def test_relu_negative_input_is_zero():
    assert relu(-5.0) == 0.0


def test_relu_positive_input_is_unchanged():
    assert relu(5.0) == 5.0


def test_relu_zero_input_is_zero():
    assert relu(0.0) == 0.0


# ---------------------------------------------------------------------------
# tanh
# ---------------------------------------------------------------------------

def test_tanh_of_zero_is_zero():
    assert tanh(0.0) == 0.0


def test_tanh_large_positive_input_approaches_one():
    assert tanh(50.0) == pytest.approx(1.0, abs=1e-6)


def test_tanh_large_negative_input_approaches_minus_one():
    assert tanh(-50.0) == pytest.approx(-1.0, abs=1e-6)


def test_tanh_matches_math_tanh():
    """Sanity check: our wrapper should agree with math.tanh exactly."""
    for x in (-3.0, -0.5, 0.0, 0.5, 3.0):
        assert tanh(x) == math.tanh(x)
