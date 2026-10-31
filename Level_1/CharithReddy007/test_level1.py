"""Quick checks. Run:  python -m pytest test_level1.py   (or: python test_level1.py)"""

import math

from similarity import cosine_similarity, dot_product, min_max_normalize, min_max_scale
from squashing import relu, sigmoid, tanh


def test_dot_product():
    assert dot_product([1, 2, 3], [4, 5, 6]) == 32


def test_cosine():
    assert math.isclose(cosine_similarity([1, 0], [1, 0]), 1.0)
    assert math.isclose(cosine_similarity([1, 0], [0, 1]), 0.0)
    assert math.isclose(cosine_similarity([1, 1], [-1, -1]), -1.0)
    assert cosine_similarity([0, 0], [1, 2]) == 0.0
    # cosine ignores length: same direction, different size -> 1
    assert math.isclose(cosine_similarity([1, 2], [10, 20]), 1.0)


def test_min_max():
    assert min_max_normalize([[0, 10], [5, 20], [10, 30]]) == [[0, 0], [0.5, 0.5], [1, 1]]
    assert min_max_scale([5], [5], [5]) == [0.0]  # constant column


def test_sigmoid_stable():
    assert sigmoid(0) == 0.5
    assert sigmoid(-1000) == 0.0
    assert sigmoid(1000) == 1.0
    for x in [-5, -1, 0.3, 4]:
        assert math.isclose(sigmoid(x), 1 / (1 + math.exp(-x)))


def test_relu_tanh():
    assert relu(-3) == 0 and relu(3) == 3
    for x in [-1000, -3, -0.2, 0, 0.2, 3, 1000]:
        assert math.isclose(tanh(x), math.tanh(x), abs_tol=1e-12)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("All tests passed.")
