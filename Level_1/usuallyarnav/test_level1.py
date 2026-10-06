"""Acceptance tests for Level 1 (Similarity and Squashing).

Place this file in Level_1/<your-github-username>/ next to similarity.py and
squashing.py, then run from anywhere:

    python -m unittest discover -s Level_1/<your-github-username> -v

or, from inside your folder:

    python -m unittest test_level1 -v

Assumed contract (rename the imports below if you chose other names):

    similarity.py
        load_songs(path)                  -> (titles, feature_names, rows)
        dot_product(a, b)                 -> float
        cosine_similarity(a, b)           -> float
        min_max_fit(rows)                 -> (mins, maxs)
        min_max_transform(row, mins, maxs) -> list[float]

    squashing.py
        sigmoid(x), relu(x), tanh(x)      -> float   (scalar in, scalar out)

These tests check mathematical properties and edge cases only. They do not
compute or reveal the song rankings.
"""

import math
import random
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from similarity import (  # noqa: E402
    cosine_similarity,
    dot_product,
    load_songs,
    min_max_fit,
    min_max_transform,
)
from squashing import relu, sigmoid, tanh  # noqa: E402

SONGS_CSV = HERE.parents[1] / "datasets" / "songs.csv"
QUERY = [124, 210, 0.78, 0.82]


def close(a, b, rel=1e-9, abs_=1e-12):
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


class TestRules(unittest.TestCase):
    def test_no_numpy(self):
        # Importing your modules must not pull in NumPy (challenge rule).
        self.assertNotIn("numpy", sys.modules)


class TestDotProduct(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(dot_product([1, 2, 3], [4, 5, 6]), 32)

    def test_length_mismatch_raises(self):
        # zip() silently truncates; a dot product of unequal vectors is a bug.
        with self.assertRaises(ValueError):
            dot_product([1, 2, 3], [4, 5])


class TestCosine(unittest.TestCase):
    def test_self_similarity_is_one(self):
        # Use isclose: the float result can be 0.9999999999999998.
        for v in ([1, 2, 3], QUERY, [0.3, 0.7, 0.1]):
            self.assertTrue(close(cosine_similarity(v, v), 1.0))

    def test_orthogonal_and_opposite(self):
        self.assertTrue(close(cosine_similarity([1, 0], [0, 1]), 0.0))
        self.assertTrue(close(cosine_similarity([1, 2], [-1, -2]), -1.0))

    def test_ignores_magnitude(self):
        a, b = [3.0, 1.0, 2.0], [1.0, 4.0, 0.5]
        scaled = [10 * x for x in a]
        self.assertTrue(close(cosine_similarity(a, b), cosine_similarity(scaled, b)))

    def test_bounded(self):
        rng = random.Random(0)
        for _ in range(500):
            a = [rng.uniform(-100, 100) for _ in range(5)]
            b = [rng.uniform(-100, 100) for _ in range(5)]
            c = cosine_similarity(a, b)
            self.assertTrue(-1 - 1e-12 <= c <= 1 + 1e-12)

    def test_zero_vector_is_handled(self):
        # Undefined direction. Either raise ValueError or return 0.0 (document
        # your choice). ZeroDivisionError or NaN is a fail.
        try:
            c = cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
        except ValueError:
            return
        self.assertFalse(math.isnan(c))
        self.assertEqual(c, 0.0)


class TestMinMax(unittest.TestCase):
    def test_fit(self):
        mins, maxs = min_max_fit([[0, 10], [5, 20], [10, 30]])
        self.assertEqual(list(mins), [0, 10])
        self.assertEqual(list(maxs), [10, 30])

    def test_training_rows_land_in_unit_interval(self):
        rows = [[0, 10], [5, 20], [10, 30]]
        mins, maxs = min_max_fit(rows)
        out = [min_max_transform(r, mins, maxs) for r in rows]
        self.assertEqual(out[0], [0.0, 0.0])
        self.assertEqual(out[2], [1.0, 1.0])
        self.assertTrue(close(out[1][0], 0.5) and close(out[1][1], 0.5))

    def test_query_uses_fitted_numbers_without_clipping(self):
        # "Scale your song using those same numbers": the query never changes
        # mins/maxs, and out-of-range values are kept, not clipped
        # (same default as scikit-learn's MinMaxScaler(clip=False)).
        mins, maxs = min_max_fit([[0], [10]])
        self.assertTrue(close(min_max_transform([20], mins, maxs)[0], 2.0))
        self.assertTrue(close(min_max_transform([-5], mins, maxs)[0], -0.5))

    def test_constant_column_does_not_divide_by_zero(self):
        rows = [[1, 5], [2, 5], [3, 5]]
        mins, maxs = min_max_fit(rows)
        col = [min_max_transform(r, mins, maxs)[1] for r in rows]
        self.assertTrue(all(math.isfinite(v) for v in col))
        self.assertEqual(len(set(col)), 1)  # recommended value: 0.0

    def test_does_not_mutate_inputs(self):
        rows = [[0.0, 10.0], [10.0, 30.0]]
        snapshot = [list(r) for r in rows]
        mins, maxs = min_max_fit(rows)
        min_max_transform(rows[0], mins, maxs)
        self.assertEqual(rows, snapshot)


@unittest.skipUnless(SONGS_CSV.exists(), f"dataset not found at {SONGS_CSV}")
class TestSongsData(unittest.TestCase):
    def test_load_shape_and_header(self):
        titles, names, rows = load_songs(SONGS_CSV)
        self.assertEqual(len(titles), 10)
        self.assertEqual(len(rows), 10)
        # Catches leftover '\r' / '\n' from the file's CRLF line endings.
        self.assertEqual(
            list(names), ["tempo_bpm", "duration_sec", "energy", "danceability"]
        )
        self.assertTrue(all(isinstance(t, str) and t == t.strip() for t in titles))
        self.assertTrue(all(len(r) == 4 for r in rows))
        self.assertTrue(all(isinstance(v, float) for r in rows for v in r))

    def test_query_normalization_and_column_order(self):
        # Fit on the 10 songs only, then transform the query with those numbers.
        _, _, rows = load_songs(SONGS_CSV)
        mins, maxs = min_max_fit(rows)
        q = min_max_transform(QUERY, mins, maxs)
        expected = [46 / 92, 34 / 124, 0.58 / 0.75, 0.57 / 0.63]
        for got, exp in zip(q, expected):
            self.assertTrue(close(got, exp, rel=1e-6))


class TestSigmoid(unittest.TestCase):
    def test_center(self):
        self.assertEqual(sigmoid(0), 0.5)

    def test_large_negative_does_not_crash(self):
        # The naive formula raises OverflowError for x < -709.78.
        for x in (-709.79, -710, -1000, -1e6):
            s = sigmoid(x)
            self.assertTrue(0.0 <= s < 1e-300, (x, s))

    def test_large_positive(self):
        for x in (40, 709.79, 1000, 1e6):
            self.assertTrue(close(sigmoid(x), 1.0))

    def test_infinities(self):
        self.assertEqual(sigmoid(float("inf")), 1.0)
        self.assertEqual(sigmoid(float("-inf")), 0.0)

    def test_symmetry(self):
        for i in range(-300, 301):
            x = i / 10
            self.assertTrue(close(sigmoid(-x), 1 - sigmoid(x), abs_=1e-15))

    def test_monotonic(self):
        xs = [i / 100 for i in range(-5000, 5001)]
        ys = [sigmoid(x) for x in xs]
        self.assertTrue(all(a <= b for a, b in zip(ys, ys[1:])))

    def test_negative_tail_keeps_precision(self):
        # Rejects fixes that return exactly 0.0 long before underflow, e.g.
        # 0.5 * (1 + tanh(x / 2)). Matters in Level 2: log(sigmoid(x)) of 0.0
        # is -inf in your loss.
        for x in (-40, -100, -500):
            exact = math.exp(x) / (1 + math.exp(x))
            self.assertTrue(close(sigmoid(x), exact, rel=1e-9, abs_=0.0), x)


class TestTanh(unittest.TestCase):
    def test_matches_math_tanh(self):
        xs = [i / 10 for i in range(-200, 201)] + [1e-12, -1e-12, 1e-8, 0.5, -3.7]
        for x in xs:
            self.assertTrue(close(tanh(x), math.tanh(x)), x)

    def test_zero_and_odd_symmetry(self):
        self.assertEqual(tanh(0), 0.0)
        for i in range(1, 300):
            x = i / 10
            self.assertTrue(close(tanh(-x), -tanh(x), abs_=1e-15))

    def test_large_inputs_do_not_crash(self):
        # The naive (e^x - e^-x)/(e^x + e^-x) overflows at |x| >= 710.
        for x in (710, 1000, 1e6):
            self.assertEqual(tanh(x), 1.0)
            self.assertEqual(tanh(-x), -1.0)

    def test_infinities(self):
        self.assertEqual(tanh(float("inf")), 1.0)
        self.assertEqual(tanh(float("-inf")), -1.0)


class TestRelu(unittest.TestCase):
    def test_values(self):
        self.assertEqual(relu(-3), 0)
        self.assertEqual(relu(0), 0)
        self.assertEqual(relu(2.5), 2.5)
        self.assertEqual(relu(1e308), 1e308)
        self.assertEqual(relu(float("-inf")), 0)
        self.assertEqual(relu(float("inf")), float("inf"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
