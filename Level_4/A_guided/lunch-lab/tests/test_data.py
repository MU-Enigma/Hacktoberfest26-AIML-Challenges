import numpy as np
import pytest

from lunchlab.data import load_csv, train_test_split


def write_csv(tmp_path, text):
    path = tmp_path / "data.csv"
    path.write_text(text)
    return str(path)


def test_real_dataset_loads():
    X, y, names = load_csv("data/lunches.csv")
    assert X.shape == (500, 4)
    assert set(y.tolist()) == {0, 1}
    assert names == ["queue_length", "plate_waste_fraction", "menu_board_kcal", "handwriting_neatness"]


def test_split_keeps_rows_and_labels_aligned():
    y = np.arange(30)
    X = (y * 10).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=3)
    assert np.array_equal(X_train[:, 0], y_train * 10)
    assert np.array_equal(X_test[:, 0], y_test * 10)


def test_split_same_seed_is_reproducible():
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20)
    a = train_test_split(X, y, seed=7)
    b = train_test_split(X, y, seed=7)
    for first, second in zip(a, b):
        assert np.array_equal(first, second)


def test_split_different_seeds_differ():
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20)
    a = train_test_split(X, y, seed=1)
    b = train_test_split(X, y, seed=2)
    assert not np.array_equal(a[3], b[3])


@pytest.mark.parametrize("bad", [0, 1, -0.1, 1.5])
def test_split_invalid_test_size_raises(bad):
    with pytest.raises(ValueError):
        train_test_split(np.zeros((10, 1)), np.zeros(10), test_size=bad)


def test_split_length_mismatch_raises():
    with pytest.raises(ValueError):
        train_test_split(np.zeros((10, 1)), np.zeros(9))
