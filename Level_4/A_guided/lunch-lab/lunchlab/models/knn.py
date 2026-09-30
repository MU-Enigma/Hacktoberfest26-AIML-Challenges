"""k-nearest-neighbours classifier for binary labels (0 and 1)."""

from typing import Any, Dict, Optional

import numpy as np


class KNNClassifier:
    """Classify a sample by a majority vote of its k nearest training samples.

    Distance is Euclidean. If the vote is tied (only possible when k is even),
    the label of the single nearest neighbour wins. If k is larger than the
    number of training samples, all training samples vote.
    """

    def __init__(self, k: int = 5) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self.X_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """Store the training data. X has shape (N, D) and y has shape (N,)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("expected a non-empty 2D array of shape (N, D)")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self.X_ = X.copy()
        self.y_ = y.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict 0 or 1 for each row of X, shape (B, D).

        Returns:
            Int array of shape (B,).
        """
        if self.X_ is None:
            raise RuntimeError("model is not fitted, call fit first")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.X_.shape[1]:
            raise ValueError(f"expected X of shape (B, {self.X_.shape[1]})")

        diff = X[:, None, :] - self.X_[None, :, :]
        distances = np.sqrt((diff**2).sum(axis=2))  # shape (B, N)

        k = min(self.k, len(self.X_))
        nearest = np.argsort(distances, axis=1, kind="stable")[:, :k]
        labels = self.y_[nearest]  # shape (B, k), sorted nearest first
        positives = labels.sum(axis=1)

        predictions = (2 * positives >= k).astype(int)
        return predictions

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly description of the fitted model."""
        if self.X_ is None:
            raise RuntimeError("model is not fitted, call fit first")
        return {"type": "knn", "k": self.k, "X": self.X_.tolist(), "y": self.y_.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KNNClassifier":
        """Rebuild a fitted model from the output of to_dict."""
        model = cls(k=d["k"])
        model.X_ = np.array(d["X"], dtype=float)
        model.y_ = np.array(d["y"], dtype=int)
        return model
