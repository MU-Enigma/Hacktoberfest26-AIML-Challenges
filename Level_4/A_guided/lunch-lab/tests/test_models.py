import json

import numpy as np
import pytest

from lunchlab.models import model_from_dict
from lunchlab.models.knn import KNNClassifier
from lunchlab.models.logistic import LogisticRegression, binary_cross_entropy, sigmoid

# ---------- sigmoid and loss ----------


def test_sigmoid_known_values():
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)
    assert sigmoid(np.array([2.0]))[0] == pytest.approx(1 / (1 + np.exp(-2.0)))


def test_sigmoid_is_stable_for_large_inputs(recwarn):
    out = sigmoid(np.array([-1000.0, 1000.0]))
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(1.0)
    assert not np.isnan(out).any()
    assert len(recwarn) == 0  # no overflow warnings


def test_sigmoid_keeps_shape():
    assert sigmoid(np.zeros((3, 2))).shape == (3, 2)


def test_binary_cross_entropy_handles_exact_zero_and_one():
    y = np.array([1.0, 0.0])
    p = np.array([0.0, 1.0])
    assert np.isfinite(binary_cross_entropy(y, p))


# ---------- logistic regression ----------

X_SEP = np.arange(6, dtype=float).reshape(-1, 1)
Y_SEP = np.array([0, 0, 0, 1, 1, 1])


def test_logistic_learns_separable_data():
    model = LogisticRegression(lr=0.5, epochs=2000, tol=0).fit(X_SEP, Y_SEP)
    assert model.predict(X_SEP).tolist() == Y_SEP.tolist()


def test_logistic_first_update_matches_the_gradient_formula():
    X = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, 0.5]])
    y = np.array([1, 0, 1])
    model = LogisticRegression(lr=0.1, epochs=1, tol=0).fit(X, y)
    error = 0.5 - y  # weights start at zero, so p = 0.5 for every sample
    assert np.allclose(model.weights_, -0.1 * (X.T @ error) / 3)
    assert model.bias_ == pytest.approx(-0.1 * error.mean())


def test_logistic_without_early_stopping_runs_all_epochs():
    model = LogisticRegression(lr=0.5, epochs=50, tol=0).fit(X_SEP, Y_SEP)
    assert model.n_iter_ == 50


def test_logistic_predict_proba_is_between_zero_and_one():
    model = LogisticRegression(lr=0.5, epochs=100, tol=0).fit(X_SEP, Y_SEP)
    proba = model.predict_proba(X_SEP)
    assert proba.shape == (6,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_logistic_rejects_bad_labels():
    with pytest.raises(ValueError):
        LogisticRegression().fit(X_SEP, np.array([0, 1, 2, 0, 1, 2]))


def test_logistic_rejects_length_mismatch():
    with pytest.raises(ValueError):
        LogisticRegression().fit(X_SEP, np.array([0, 1]))


def test_logistic_unfitted_predict_raises():
    with pytest.raises(RuntimeError):
        LogisticRegression().predict(X_SEP)


def test_logistic_wrong_feature_count_raises():
    model = LogisticRegression(epochs=5).fit(X_SEP, Y_SEP)
    with pytest.raises(ValueError):
        model.predict(np.zeros((2, 3)))


def test_logistic_dict_round_trip():
    model = LogisticRegression(lr=0.5, epochs=100, tol=0).fit(X_SEP, Y_SEP)
    rebuilt = model_from_dict(json.loads(json.dumps(model.to_dict())))
    assert isinstance(rebuilt, LogisticRegression)
    assert np.allclose(rebuilt.predict_proba(X_SEP), model.predict_proba(X_SEP))


# ---------- kNN ----------

X_KNN = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]])
Y_KNN = np.array([0, 0, 1, 1])


def test_knn_predicts_by_majority_vote():
    model = KNNClassifier(k=3).fit(X_KNN, Y_KNN)
    assert model.predict(np.array([[0.0, 0.5], [10.0, 10.5]])).tolist() == [0, 1]


def test_knn_with_k_one_copies_training_labels():
    model = KNNClassifier(k=1).fit(X_KNN, Y_KNN)
    assert model.predict(X_KNN).tolist() == Y_KNN.tolist()


def test_knn_k_larger_than_training_set_uses_all_samples():
    model = KNNClassifier(k=10).fit(np.array([[0.0], [1.0], [2.0]]), np.array([0, 0, 1]))
    assert model.predict(np.array([[2.0]])).tolist() == [0]


def test_knn_invalid_k_raises():
    with pytest.raises(ValueError):
        KNNClassifier(k=0)


def test_knn_unfitted_predict_raises():
    with pytest.raises(RuntimeError):
        KNNClassifier().predict(X_KNN)


def test_knn_wrong_feature_count_raises():
    model = KNNClassifier(k=1).fit(X_KNN, Y_KNN)
    with pytest.raises(ValueError):
        model.predict(np.zeros((2, 3)))


def test_knn_dict_round_trip():
    model = KNNClassifier(k=3).fit(X_KNN, Y_KNN)
    rebuilt = model_from_dict(json.loads(json.dumps(model.to_dict())))
    queries = np.array([[0.0, 0.2], [9.0, 9.0]])
    assert rebuilt.predict(queries).tolist() == model.predict(queries).tolist()


def test_model_from_dict_unknown_type_raises():
    with pytest.raises(ValueError):
        model_from_dict({"type": "nope"})
