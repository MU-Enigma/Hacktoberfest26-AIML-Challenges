"""Saving and loading a trained model together with its scaler."""

import json
from typing import List, Tuple

from lunchlab.models import model_from_dict
from lunchlab.preprocessing import scaler_from_dict


def save_bundle(path: str, model, scaler, feature_names: List[str]) -> None:
    """Write the model, the scaler and the feature names to a JSON file.

    The file has the keys "model", "scaler" and "features".
    """
    raise NotImplementedError("save_bundle is not implemented yet")


def load_bundle(path: str) -> Tuple[object, object, List[str]]:
    """Read a file written by save_bundle.

    Returns:
        model, scaler, feature_names. The loaded model and scaler give exactly the
        same predictions as the ones that were saved.
    """
    raise NotImplementedError("load_bundle is not implemented yet")
