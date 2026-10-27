"""Unit Tests for Level 1 Implementation.

Uses Python standard library 'unittest' (NO NumPy).
Validates mathematical correctness, boundary behavior, and edge cases.
"""

import math
import os
import sys
import unittest

# Ensure the current directory is on sys.path regardless of where test is run from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from similarity import (
    compute_feature_bounds,
    cosine_similarity,
    dot_product,
    load_dataset,
    min_max_scale,
    rank_songs,
    vector_magnitude,
)
from squashing import relu, sigmoid, sigmoid_naive, tanh


class TestVectorMath(unittest.TestCase):
    def test_dot_product(self):
        self.assertAlmostEqual(dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0)
        self.assertAlmostEqual(dot_product([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertAlmostEqual(dot_product([-1.0, 2.0], [3.0, -4.0]), -11.0)
        with self.assertRaises(ValueError):
            dot_product([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_vector_magnitude(self):
        self.assertAlmostEqual(vector_magnitude([3.0, 4.0]), 5.0)
        self.assertAlmostEqual(vector_magnitude([0.0, 0.0, 0.0]), 0.0)
        self.assertAlmostEqual(vector_magnitude([1.0]), 1.0)

    def test_cosine_similarity(self):
        # Parallel vectors
        self.assertAlmostEqual(cosine_similarity([1.0, 2.0], [2.0, 4.0]), 1.0)
        # Orthogonal vectors
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        # Opposite vectors
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)
        # Zero vector handling
        self.assertAlmostEqual(cosine_similarity([0.0, 0.0], [1.0, 2.0]), 0.0)

    def test_min_max_scale(self):
        self.assertAlmostEqual(min_max_scale(10.0, 10.0, 20.0), 0.0)
        self.assertAlmostEqual(min_max_scale(20.0, 10.0, 20.0), 1.0)
        self.assertAlmostEqual(min_max_scale(15.0, 10.0, 20.0), 0.5)
        # Equal min and max edge case
        self.assertAlmostEqual(min_max_scale(10.0, 10.0, 10.0), 0.0)


class TestSquashingFunctions(unittest.TestCase):
    def test_relu(self):
        self.assertEqual(relu(-5.0), 0.0)
        self.assertEqual(relu(0.0), 0.0)
        self.assertEqual(relu(4.2), 4.2)

    def test_tanh(self):
        self.assertAlmostEqual(tanh(0.0), 0.0)
        self.assertAlmostEqual(tanh(100.0), 1.0)
        self.assertAlmostEqual(tanh(-100.0), -1.0)
        self.assertTrue(-1.0 <= tanh(-2.5) <= 1.0)

    def test_sigmoid_standard_points(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5)
        self.assertAlmostEqual(sigmoid(2.0), 1.0 / (1.0 + math.exp(-2.0)))
        self.assertAlmostEqual(sigmoid(-2.0), math.exp(-2.0) / (1.0 + math.exp(-2.0)))

    def test_sigmoid_overflow_edge_cases(self):
        # Naive sigmoid raises OverflowError on large negative numbers
        with self.assertRaises(OverflowError):
            sigmoid_naive(-1000.0)

        # Robust sigmoid handles extreme values gracefully
        self.assertEqual(sigmoid(-1000.0), 0.0)
        self.assertEqual(sigmoid(1000.0), 1.0)


class TestSongRecommender(unittest.TestCase):
    def setUp(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.normpath(
            os.path.join(base_dir, "..", "..", "datasets", "songs.csv")
        )
        self.features = ["tempo_bpm", "duration_sec", "energy", "danceability"]
        self.target_song = {
            "title": "Loved Song",
            "tempo_bpm": 124.0,
            "duration_sec": 210.0,
            "energy": 0.78,
            "danceability": 0.82,
        }

    def test_dataset_loading_and_bounds(self):
        songs = load_dataset(self.csv_path, self.features)
        self.assertEqual(len(songs), 10)
        min_b, max_b = compute_feature_bounds(songs, self.features)
        self.assertEqual(min_b["tempo_bpm"], 78.0)
        self.assertEqual(max_b["tempo_bpm"], 170.0)

    def test_rankings(self):
        songs = load_dataset(self.csv_path, self.features)
        min_b, max_b = compute_feature_bounds(songs, self.features)

        # Unnormalized top is Rooftop Monsoon due to scale dominance
        unnorm = rank_songs(songs, self.target_song, self.features, normalize=False)
        self.assertEqual(unnorm[0][0], "Rooftop Monsoon")

        # Normalized top is Fresher Night
        norm = rank_songs(
            songs,
            self.target_song,
            self.features,
            normalize=True,
            min_bounds=min_b,
            max_bounds=max_b,
        )
        self.assertEqual(norm[0][0], "Fresher Night")
        self.assertEqual(norm[1][0], "Canteen Bass")


if __name__ == "__main__":
    unittest.main()
