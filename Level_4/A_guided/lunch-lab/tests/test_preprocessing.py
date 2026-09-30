import json

import numpy as np
import pytest

from lunchlab.preprocessing import MinMaxScaler, StandardScaler, scaler_from_dict


def test_minmax_uses_training_statistics_and_does_not_clip():
    scaler = MinMaxScaler().fit(np.array([[0.0], [10.0]]))
    out = scaler.transform(np.array([[20.0], [-10.0]]))
    assert np.allclose(out, [[2.0], [-1.0]])


def test_minmax_constant_column_becomes_zero():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    out = MinMaxScaler().fit_transform(X)
    assert np.allclose(out[:, 1], 0)
    assert not np.isnan(out).any()


def test_minmax_unfitted_raises():
    with pytest.raises(RuntimeError):
        MinMaxScaler().transform(np.zeros((2, 2)))


@pytest.mark.parametrize("bad", [np.array([1.0, 2.0, 3.0]), np.zeros((0, 3))])
def test_minmax_rejects_bad_shapes(bad):
    with pytest.raises(ValueError):
        MinMaxScaler().fit(bad)


@pytest.mark.parametrize("cls", [MinMaxScaler])
def test_scaler_dict_round_trip(cls):
    X = np.array([[1.0, 10.0], [2.0, 30.0], [4.0, 20.0]])
    scaler = cls().fit(X)
    d = json.loads(json.dumps(scaler.to_dict()))
    rebuilt = scaler_from_dict(d)
    assert type(rebuilt) is cls
    assert np.allclose(rebuilt.transform(X), scaler.transform(X))


def test_scaler_from_dict_unknown_type_raises():
    with pytest.raises(ValueError):
        scaler_from_dict({"type": "nope"})
