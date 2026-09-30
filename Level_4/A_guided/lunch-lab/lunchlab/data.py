"""Loading data and splitting it into train and test sets."""

import csv
from typing import List, Optional, Tuple

import numpy as np


def load_csv(
    path: str, target: Optional[str] = "good_lunch"
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """Load a CSV file that has a header row.

    Args:
        path: Path to the CSV file.
        target: Name of the label column. If None, every column is treated as a
            feature and no labels are returned.

    Returns:
        X: Float array of shape (N, D) with the feature columns, in file order.
        y: Int array of shape (N,) with the labels, or None if target is None.
        feature_names: The names of the D feature columns, in file order.

    Raises:
        ValueError: If target is given but is not a column in the file.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]

    if target is not None and target not in header:
        raise ValueError(f"target column '{target}' not found, columns are {header}")

    target_idx = header.index(target) if target is not None else None
    feature_idx = [i for i in range(len(header)) if i != target_idx]

    X = np.array([[float(row[i]) for i in feature_idx] for row in rows], dtype=float)
    y = None
    if target_idx is not None:
        y = np.array([int(row[target_idx]) for row in rows], dtype=int)
    feature_names = [header[i] for i in feature_idx]
    return X, y, feature_names


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle the data and split it into a train part and a test part.

    The number of test samples is round(N * test_size). Every sample ends up in
    exactly one of the two parts. The same seed always gives the same split.

    Args:
        X: Features, shape (N, D).
        y: Labels, shape (N,).
        test_size: Fraction of samples for the test set, strictly between 0 and 1.
        seed: Seed for the shuffle.

    Returns:
        X_train, X_test, y_train, y_test

    Raises:
        ValueError: If test_size is not strictly between 0 and 1, or if X and y
            have different lengths.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    n_test = int(round(len(X) * test_size))
    test_idx, train_idx = order[: n_test + 1], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
