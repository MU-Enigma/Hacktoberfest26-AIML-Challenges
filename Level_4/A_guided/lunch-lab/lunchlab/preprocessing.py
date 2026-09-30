"""Feature scaling. Fit on the training data only, then apply to any data."""

from typing import Any, Dict, Optional

import numpy as np


def _as_2d(X: Any) -> np.ndarray:
    """Convert to a float array and check that it is non-empty and 2D."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("expected a non-empty 2D array of shape (N, D)")
    return X


def _check_fitted(value: Optional[np.ndarray]) -> None:
    if value is None:
        raise RuntimeError("scaler is not fitted, call fit first")


class MinMaxScaler:
    """Scale each feature (column) to the range [0, 1] using the training min and max.

    A constant column (max equals min) is mapped to all zeros. Data outside the
    training range is NOT clipped, so transformed values can fall outside [0, 1].
    """

    def __init__(self) -> None:
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """Learn the per-column minimum and maximum from X, shape (N, D)."""
        X = _as_2d(X)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale X, shape (B, D), using the min and max learned in fit.

        Returns:
            Array of shape (B, D). Each column is (x - min) / (max - min), and a
            column with max == min becomes all zeros.
        """
        _check_fitted(self.min_)
        X = _as_2d(X)
        span = np.where(self.max_ == 0, 1.0, self.max_)
        return (X - self.min_) / span

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X, then transform X."""
        return self.fit(X).transform(X)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly description of the fitted scaler."""
        _check_fitted(self.min_)
        return {"type": "minmax", "min": self.min_.tolist(), "max": self.max_.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MinMaxScaler":
        """Rebuild a fitted scaler from the output of to_dict."""
        scaler = cls()
        scaler.min_ = np.array(d["min"], dtype=float)
        scaler.max_ = np.array(d["max"], dtype=float)
        return scaler


class StandardScaler:
    """Shift and scale each feature (column) to mean 0 and standard deviation 1.

    Uses the population standard deviation (ddof=0). A constant column is mapped
    to all zeros.
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Learn the per-column mean and standard deviation from X, shape (N, D)."""
        raise NotImplementedError("StandardScaler.fit is not implemented yet")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize X, shape (B, D), using the statistics learned in fit."""
        raise NotImplementedError("StandardScaler.transform is not implemented yet")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X, then transform X."""
        return self.fit(X).transform(X)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly description of the fitted scaler."""
        raise NotImplementedError("StandardScaler.to_dict is not implemented yet")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StandardScaler":
        """Rebuild a fitted scaler from the output of to_dict."""
        raise NotImplementedError("StandardScaler.from_dict is not implemented yet")


SCALERS = {"minmax": MinMaxScaler, "standard": StandardScaler}


def scaler_from_dict(d: Dict[str, Any]):
    """Rebuild a scaler of the right kind from the output of its to_dict."""
    if d["type"] not in SCALERS:
        raise ValueError(f"unknown scaler type '{d['type']}'")
    return SCALERS[d["type"]].from_dict(d)
