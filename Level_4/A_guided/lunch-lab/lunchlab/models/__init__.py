"""The available models, and a helper to rebuild one from a saved description."""

from typing import Any, Dict

from lunchlab.models.knn import KNNClassifier
from lunchlab.models.logistic import LogisticRegression

MODELS = {"logistic": LogisticRegression, "knn": KNNClassifier}


def model_from_dict(d: Dict[str, Any]):
    """Rebuild a fitted model of the right kind from the output of its to_dict."""
    if d["type"] not in MODELS:
        raise ValueError(f"unknown model type '{d['type']}'")
    return MODELS[d["type"]].from_dict(d)
