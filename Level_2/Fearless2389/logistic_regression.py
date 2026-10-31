"""
Level 2: logistic regression trained with batch gradient descent.

Task: predict `good_lunch` (1 = good, 0 = not good) from four numeric columns of
datasets/binary_classification.csv. Everything that "learns" is written by hand
with NumPy: the sigmoid, the binary cross-entropy loss, its gradient, and the
update step. No scikit-learn.

Run from anywhere:
    python logistic_regression.py

What it does:
    1. loads the CSV and makes its own 80/20 train/test split (seed 42)
    2. standardizes the features using TRAIN statistics only
    3. trains with three learning rates and saves loss_curves.png
    4. reports test accuracy and time to convergence for each
    5. prints the error analysis used in analysis.md
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Settings (all in one place so the run is easy to reproduce)
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent  # folder this script lives in
DATA_PATH = HERE.parent.parent / "datasets" / "binary_classification.csv"
PLOT_PATH = HERE / "loss_curves.png"

FEATURES = ["queue_length", "plate_waste_fraction", "menu_board_kcal", "handwriting_neatness"]
LABEL = "good_lunch"

SEED = 42  # fixed seed -> same split every run
TEST_FRACTION = 0.2  # 20% of rows are held out for testing

# Convergence criterion: stop when the training loss changes by less than TOL
# between two consecutive iterations, or give up after MAX_ITERS iterations.
TOL = 1e-6
MAX_ITERS = 10_000

# The three learning rates compared (predictions for each are in README.md).
LEARNING_RATES = [0.001, 0.5, 25.0]
BEST_LR = 0.5  # the one used for the final model and the error analysis


# ---------------------------------------------------------------------------
# 1. Data: load, split, standardize
# ---------------------------------------------------------------------------


def load_data(path, features=FEATURES):
    """Read the CSV and return (X, y): X is an (n, d) float array, y is an (n,) 0/1 array."""
    df = pd.read_csv(path)
    X = df[features].to_numpy(dtype=float)
    y = df[LABEL].to_numpy(dtype=float)
    return X, y


def train_test_split(X, y, test_fraction=TEST_FRACTION, seed=SEED):
    """Shuffle the row indices with a seeded generator and cut them into train and test."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))  # e.g. [312, 7, 455, ...]
    n_test = int(round(len(y) * test_fraction))
    test_idx, train_idx = order[:n_test], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], train_idx, test_idx


def fit_standardizer(X_train):
    """Compute the mean and standard deviation of each column of the TRAINING data only.

    Why train only: the test set must play the role of "lunches we have never seen".
    If its rows helped compute the mean/std, information about them would leak into
    the model before we test it.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0  # guard: a constant column would otherwise divide by zero
    return mean, std


def standardize(X, mean, std):
    """Rescale every column to roughly mean 0, std 1 (z-score), using the given statistics.

    Why at all: the raw columns live on very different scales (menu_board_kcal is
    around 2000, plate_waste_fraction is between 0 and 1). Without rescaling, one
    learning rate would be far too big for the kcal weight and far too small for the
    fraction weight, and gradient descent would crawl or blow up.
    """
    return (X - mean) / std


# ---------------------------------------------------------------------------
# 2. The model: sigmoid, loss, gradient, update
# ---------------------------------------------------------------------------


def sigmoid(z):
    """Squash any real number into (0, 1): sigmoid(z) = 1 / (1 + e^(-z))."""
    z = np.clip(z, -500, 500)  # avoid overflow warnings in exp for extreme z
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(X, w, b):
    """Probability that each row is a good lunch: sigmoid(X . w + b)."""
    return sigmoid(X @ w + b)


def bce_loss(y, p, eps=1e-12):
    """Mean binary cross-entropy: -mean( y*log(p) + (1-y)*log(1-p) ).

    It is small when we give high probability to the true label and grows without
    limit when we are confidently wrong. eps keeps log() away from log(0).
    """
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def gradients(X, y, p):
    """Gradient of the mean BCE loss with respect to w and b.

    For logistic regression the calculus simplifies nicely:
        dL/dw = X^T (p - y) / n      dL/db = mean(p - y)
    (p - y) is the per-row error: positive when we over-predicted "good".
    """
    n = len(y)
    error = p - y
    grad_w = X.T @ error / n
    grad_b = error.mean()
    return grad_w, grad_b


def train(X, y, lr, max_iters=MAX_ITERS, tol=TOL):
    """Batch gradient descent: every iteration uses ALL training rows.

    Returns a dict with the learned weights, the loss at every iteration,
    how many iterations ran, whether the convergence criterion was met,
    and the wall-clock time in seconds.
    """
    w = np.zeros(X.shape[1])  # start from "know nothing": every p = 0.5
    b = 0.0
    losses = []
    converged = False
    steps = 0  # counts how many updates we actually made

    start = time.perf_counter()
    for it in range(max_iters):
        p = predict_proba(X, w, b)  # forward pass
        loss = bce_loss(y, p)
        losses.append(loss)

        # Convergence check: has the loss stopped moving?
        if it > 0 and abs(losses[-2] - loss) < tol:
            converged = True
            break

        grad_w, grad_b = gradients(X, y, p)  # which way is "uphill"
        w = w - lr * grad_w  # step downhill
        b = b - lr * grad_b
        steps += 1

    if not converged:
        # We hit the iteration cap: also record the loss of the FINAL weights,
        # so losses[-1] always describes the model we return.
        losses.append(bce_loss(y, predict_proba(X, w, b)))
    seconds = time.perf_counter() - start

    return {
        "w": w,
        "b": b,
        "losses": np.array(losses),
        "iters": steps,  # number of update steps taken
        "converged": converged,
        "seconds": seconds,
    }


def accuracy(y_true, p, threshold=0.5):
    """Fraction of rows where (p >= threshold) matches the true 0/1 label."""
    return np.mean((p >= threshold).astype(float) == y_true)


# ---------------------------------------------------------------------------
# 3. Experiments
# ---------------------------------------------------------------------------


def prepare(features=FEATURES):
    """Load, split and standardize. Returns everything later steps need."""
    X, y = load_data(DATA_PATH, features)
    X_tr, X_te, y_tr, y_te, tr_idx, te_idx = train_test_split(X, y)
    mean, std = fit_standardizer(X_tr)
    return {
        "X_tr_raw": X_tr,
        "X_te_raw": X_te,
        "X_tr": standardize(X_tr, mean, std),
        "X_te": standardize(X_te, mean, std),
        "y_tr": y_tr,
        "y_te": y_te,
        "test_idx": te_idx,
        "features": features,
    }


def compare_learning_rates(data):
    """Train once per learning rate, print a results table, return the runs."""
    runs = {}
    print(f"Convergence criterion: |loss_t - loss_(t-1)| < {TOL} or {MAX_ITERS} iterations\n")
    print(f"{'lr':>7} {'iters':>6} {'converged':>9} {'time (ms)':>9} {'train loss':>10} {'test acc':>8}")
    for lr in LEARNING_RATES:
        run = train(data["X_tr"], data["y_tr"], lr)
        run["test_acc"] = accuracy(data["y_te"], predict_proba(data["X_te"], run["w"], run["b"]))
        runs[lr] = run
        print(
            f"{lr:>7} {run['iters']:>6} {str(run['converged']):>9} "
            f"{run['seconds'] * 1000:>9.1f} {run['losses'][-1]:>10.4f} {run['test_acc']:>8.3f}"
        )
    return runs


def tolerance_check(data, run):
    """Did TOL = 1e-6 stop too early? Retrain the best lr with a much stricter tolerance and compare."""
    strict = train(data["X_tr"], data["y_tr"], BEST_LR, max_iters=100_000, tol=1e-12)
    strict_acc = accuracy(data["y_te"], predict_proba(data["X_te"], strict["w"], strict["b"]))
    print(f"\nStricter tolerance 1e-12 at lr {BEST_LR}: {strict['iters']} updates, "
          f"train loss only {run['losses'][-1] - strict['losses'][-1]:.1e} lower, test acc {strict_acc:.3f}")


def largest_curvature(X, w, b):
    """Largest eigenvalue of the loss's Hessian (its "steepest bend") at weights (w, b).

    Hessian of mean BCE = X1^T diag(p(1-p)) X1 / n, where X1 is X with a column of 1s
    for the bias. Gradient descent is only stable when lr < 2 / (this number), which
    explains why lr = 25 zig-zags (see README).
    """
    X1 = np.column_stack([X, np.ones(len(X))])
    p = predict_proba(X, w, b)
    hessian = X1.T @ (X1 * (p * (1 - p))[:, None]) / len(X)
    return np.linalg.eigvalsh(hessian).max()


def stability_check(data, run):
    """Print the largest stable learning rate at the start (w = 0) and at the learned weights."""
    for where, w, b in (("start (w = 0)", np.zeros(data["X_tr"].shape[1]), 0.0), ("learned weights", run["w"], run["b"])):
        lam = largest_curvature(data["X_tr"], w, b)
        print(f"Largest curvature at {where}: {lam:.3f} -> stable only if lr < {2 / lam:.1f}")


def plot_loss_curves(runs, path=PLOT_PATH):
    """Two panels: the whole run (log x-axis) and a zoom on the first 40 updates."""
    fig, (ax_all, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = {0.001: "lr = 0.001 (tiny)", 0.5: "lr = 0.5", 25.0: "lr = 25 (too large)"}
    for lr, run in runs.items():
        steps = np.arange(len(run["losses"]))  # 0 = before any update, 1 = after one update, ...
        # A log axis cannot show 0, so the left panel is shifted by one.
        ax_all.plot(steps + 1, run["losses"], label=labels.get(lr, f"lr = {lr}"))
        ax_zoom.plot(steps[:40], run["losses"][:40], marker=".", label=labels.get(lr, f"lr = {lr}"))

    ax_all.set_xscale("log")
    ax_all.set_title("Training loss, whole run (log x-axis)")
    ax_all.set_xlabel("updates done + 1 (log scale)")
    ax_zoom.set_title("First 40 updates")
    ax_zoom.set_xlabel("updates done")
    for ax in (ax_all, ax_zoom):
        ax.set_ylabel("binary cross-entropy (train)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\nSaved {path.name}")


# ---------------------------------------------------------------------------
# 4. Error analysis (the numbers quoted in analysis.md)
# ---------------------------------------------------------------------------


def most_confident_mistakes(data, run, k=3):
    """Return test positions of the k mistakes where the model was MOST sure of the wrong answer.

    "Most sure" = predicted probability furthest from 0.5. Picking these (instead of
    hand-picking) keeps the choice of examples honest.
    """
    p = predict_proba(data["X_te"], run["w"], run["b"])
    wrong = np.where((p >= 0.5) != data["y_te"])[0]
    confidence = np.abs(p[wrong] - 0.5)
    return wrong[np.argsort(-confidence)[:k]], p


def show_examples(data, run, positions, p):
    """Print raw features, true label, predicted probability and each feature's push on the logit."""
    w = run["w"]
    for pos in positions:
        raw = data["X_te_raw"][pos]
        z = data["X_te"][pos]
        print(f"\n  CSV row {data['test_idx'][pos] + 2} (line in file, header = line 1)")
        print(f"  true good_lunch = {int(data['y_te'][pos])}, predicted P(good) = {p[pos]:.3f}")
        for name, value, contribution in zip(data["features"], raw, w * z):
            print(f"    {name:<22} = {value:<8g} pushes logit by {contribution:+.3f}")


def error_analysis(data, run):
    """Show the 3 most-confident test mistakes, then test the hypothesis by dropping one feature."""
    print("\n=== Error analysis ===")
    print("Learned weights (on standardized features):")
    for name, weight in zip(data["features"], run["w"]):
        print(f"  {name:<22} {weight:+.3f}")
    print(f"  {'bias':<22} {run['b']:+.3f}")

    # How strongly does each feature move with the label? (train split vs test split)
    print("\nCorrelation of each feature with good_lunch (train vs test split):")
    for j, name in enumerate(data["features"]):
        r_tr = np.corrcoef(data["X_tr_raw"][:, j], data["y_tr"])[0, 1]
        r_te = np.corrcoef(data["X_te_raw"][:, j], data["y_te"])[0, 1]
        print(f"  {name:<22} train {r_tr:+.3f}   test {r_te:+.3f}")

    positions, p = most_confident_mistakes(data, run)
    print("\nThe 3 most-confident mistakes on the test set (all 4 features):")
    show_examples(data, run, positions, p)

    # The test: drop handwriting_neatness, keep everything else identical, retrain.
    kept = [f for f in FEATURES if f != "handwriting_neatness"]
    small = prepare(kept)
    small_run = train(small["X_tr"], small["y_tr"], BEST_LR)
    p_small = predict_proba(small["X_te"], small_run["w"], small_run["b"])

    p_full = predict_proba(data["X_te"], run["w"], run["b"])
    print("\nSame split, same lr, WITHOUT handwriting_neatness:")
    print(f"  iterations {small_run['iters']}, converged {small_run['converged']}")
    print(f"  test accuracy: {accuracy(data['y_te'], p_full):.3f} (4 features) -> "
          f"{accuracy(small['y_te'], p_small):.3f} (3 features)")
    print(f"  test loss:     {bce_loss(data['y_te'], p_full):.4f} (4 features) -> "
          f"{bce_loss(small['y_te'], p_small):.4f} (3 features)")
    print("  the same 3 examples, P(good) before -> after:")
    for pos in positions:
        print(f"    CSV row {data['test_idx'][pos] + 2}: true {int(data['y_te'][pos])}, "
              f"{p_full[pos]:.3f} -> {p_small[pos]:.3f}")


def split_sensitivity(n_seeds=20):
    """Sanity check: is seed 42 a typical split? Repeat the 4- vs 3-feature comparison on 20 splits.

    Not part of the single "small change" test; it only tells us how much one
    100-row test set can be trusted.
    """
    print(f"\n=== Sanity check over {n_seeds} different splits (lr {BEST_LR}) ===")
    X_all, y_all = load_data(DATA_PATH)
    acc_full, acc_small, loss_full, loss_small = [], [], [], []
    for seed in range(n_seeds):
        for cols, accs, losses in (([0, 1, 2, 3], acc_full, loss_full), ([0, 1, 2], acc_small, loss_small)):
            X_tr, X_te, y_tr, y_te, _, _ = train_test_split(X_all[:, cols], y_all, seed=seed)
            mean, std = fit_standardizer(X_tr)
            run = train(standardize(X_tr, mean, std), y_tr, BEST_LR)
            p = predict_proba(standardize(X_te, mean, std), run["w"], run["b"])
            accs.append(accuracy(y_te, p))
            losses.append(bce_loss(y_te, p))
    acc_full, acc_small = np.array(acc_full), np.array(acc_small)
    loss_full, loss_small = np.array(loss_full), np.array(loss_small)
    print(f"  4 features: mean test acc {acc_full.mean():.3f} (min {acc_full.min():.2f}, max {acc_full.max():.2f})")
    print(f"  3 features: mean test acc {acc_small.mean():.3f} (min {acc_small.min():.2f}, max {acc_small.max():.2f})")
    print(f"  dropping neatness lowered test loss on {np.sum(loss_small < loss_full)} of {n_seeds} splits, "
          f"raised accuracy on {np.sum(acc_small > acc_full)}, tied on {np.sum(acc_small == acc_full)}")


def main():
    data = prepare()
    print(f"Train rows: {len(data['y_tr'])}, test rows: {len(data['y_te'])}")
    baseline = 1 - data["y_te"].mean()  # accuracy of always answering "not good"
    print(f"Baseline test accuracy (always predict 'not good'): {baseline:.3f}\n")
    runs = compare_learning_rates(data)
    tolerance_check(data, runs[BEST_LR])
    stability_check(data, runs[BEST_LR])
    plot_loss_curves(runs)
    error_analysis(data, runs[BEST_LR])
    split_sensitivity()


if __name__ == "__main__":
    main()
