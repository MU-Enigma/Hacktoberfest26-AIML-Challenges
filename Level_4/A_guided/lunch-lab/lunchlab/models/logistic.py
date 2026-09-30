"""Logistic regression trained with gradient descent, using only NumPy."""

from typing import Any, Dict, List, Optional

import numpy as np

EPS = 1e-12


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid, 1 / (1 + exp(-z)), applied element-wise.

    Must not overflow for inputs like -1000 or 1000.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def binary_cross_entropy(y: np.ndarray, p: np.ndarray) -> float:
    """Mean binary cross-entropy: -mean(y*log(p) + (1-y)*log(1-p)).

    Probabilities are clipped to [1e-12, 1 - 1e-12] so the log never sees 0.
    """
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


class LogisticRegression:
    """Binary classifier: P(y = 1 | x) = sigmoid(w . x + b).

    Trained with batch gradient descent on the mean binary cross-entropy.
    Training stops early once the loss changes by less than `tol` between two
    epochs. With tol=0 it always runs all `epochs`.
    """

    def __init__(self, lr: float = 0.1, epochs: int = 1000, tol: float = 1e-6) -> None:
        self.lr = lr
        self.epochs = epochs
        self.tol = tol
        self.weights_: Optional[np.ndarray] = None
        self.bias_: float = 0.0
        self.loss_history_: List[float] = []

    @property
    def n_iter_(self) -> int:
        """Number of epochs actually run in the last call to fit."""
        return len(self.loss_history_)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Train on X, shape (N, D), and binary labels y, shape (N,).

        The gradients of the mean loss are:
            dL/dw = X^T (p - y) / N
            dL/db = mean(p - y)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("expected a non-empty 2D array of shape (N, D)")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("labels must be 0 or 1")

        n, d = X.shape
        self.weights_ = np.zeros(d)
        self.bias_ = 0.0
        self.loss_history_ = []

        for _ in range(self.epochs):
            p = sigmoid(X @ self.weights_ + self.bias_)
            self.loss_history_.append(binary_cross_entropy(y, p))

            error = p - y
            self.weights_ -= self.lr * (X.T @ error) / n
            self.bias_ -= self.lr * error.mean()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(y = 1) for each row of X, shape (B, D). Result has shape (B,)."""
        if self.weights_ is None:
            raise RuntimeError("model is not fitted, call fit first")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.weights_):
            raise ValueError(f"expected X of shape (B, {len(self.weights_)})")
        return sigmoid(X @ self.weights_ + self.bias_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 where P(y = 1) >= 0.5, else 0. Result has shape (B,)."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly description of the fitted model."""
        if self.weights_ is None:
            raise RuntimeError("model is not fitted, call fit first")
        return {
            "type": "logistic",
            "lr": self.lr,
            "epochs": self.epochs,
            "tol": self.tol,
            "weights": self.weights_.tolist(),
            "bias": float(self.bias_),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LogisticRegression":
        """Rebuild a fitted model from the output of to_dict."""
        model = cls(lr=d["lr"], epochs=d["epochs"], tol=d["tol"])
        model.weights_ = np.array(d["weights"], dtype=float)
        model.bias_ = float(d["bias"])
        return model
