"""Level 2: logistic regression trained with gradient descent, written by hand.

Predicts `good_lunch` (1 = good) from the other columns of
datasets/binary_classification.csv. NumPy and matplotlib only: the gradient
and the update step are written out below, no scikit-learn.

Run from the repo root:
    python Level_2/<your-github-username>/logistic_regression.py

It prints the test accuracy, convergence details for three learning rates,
the misclassified test lunches, and a feature-dropping experiment, and it
saves loss_curves.png next to this file.
"""

import csv
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")  # save the plot to a file; no window needed
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "..", "..", "datasets", "binary_classification.csv")
TARGET = "good_lunch"

SEED = 42                     # fixed seed, so the split is the same every run
TEST_FRACTION = 0.2           # 400 lunches to train on, 100 to test on
LEARNING_RATES = [0.001, 0.1, 50.0]   # too small, sensible, too big
MAIN_LR = 0.1                 # the model we report and analyse
MAX_EPOCHS = 20000
GRAD_TOL = 1e-4               # convergence: gradient length below this


# ---------- data ----------

def load_data(path):
    """Read the CSV. Returns (feature_names, X, y) as NumPy arrays."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path} has no rows")
    if TARGET not in rows[0]:
        sys.exit(f"No '{TARGET}' column in {path}. Columns: {list(rows[0])}")

    features = [c for c in rows[0] if c != TARGET]
    try:
        X = np.array([[float(r[c]) for c in features] for r in rows])
    except ValueError:
        bad = [c for c in features if not all(_is_number(r[c]) for r in rows)]
        sys.exit(f"Non-numeric feature column(s): {bad}. They need encoding first.")
    y = np.array([float(r[TARGET]) for r in rows])
    return features, X, y


def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def train_test_split(X, y, test_fraction=TEST_FRACTION, seed=SEED):
    """Shuffle the row order with a fixed seed, then cut off the test set."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    n_test = int(round(len(y) * test_fraction))
    test, train = order[:n_test], order[n_test:]
    return X[train], X[test], y[train], y[test], train, test


def fit_scaler(X_train):
    """Mean and spread of each feature, from the TRAINING set only.

    Using test data here would leak information about the test set into
    training, so the test accuracy would look better than it really is.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0       # a constant column: leave it unscaled
    return mean, std


def scale(X, mean, std):
    return (X - mean) / std


# ---------- the model ----------

def sigmoid(z):
    """Stable sigmoid (same trick as Level 1): never computes e^(big number)."""
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def predict_proba(X, w, b):
    """Probability of a good lunch: sigmoid of a weighted sum of features."""
    return sigmoid(X @ w + b)


def log_loss(p, y):
    """Binary cross-entropy: punishes confident wrong answers heavily."""
    p = np.clip(p, 1e-12, 1 - 1e-12)   # log(0) would be -infinity
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def gradients(X, y, w, b):
    """Slope of the loss with respect to each weight and the bias.

    For logistic regression with cross-entropy, the chain rule collapses to:
        dL/dw = X^T (p - y) / n
        dL/db = mean(p - y)
    (p - y) is just "how wrong each prediction was".
    """
    error = predict_proba(X, w, b) - y
    return X.T @ error / len(y), float(error.mean())


def train(X, y, lr, max_epochs=MAX_EPOCHS, grad_tol=GRAD_TOL):
    """Full-batch gradient descent from all-zero weights.

    Convergence criterion: stop when the gradient's length (all weights and
    the bias together) drops below grad_tol, i.e. the loss surface is flat
    and further steps would barely change anything. We use the gradient
    rather than "loss stopped changing" because a tiny learning rate makes
    the loss change slowly even when we are far from the minimum, which
    would falsely look converged.
    """
    w, b = np.zeros(X.shape[1]), 0.0
    losses = []
    start = time.perf_counter()
    converged = False
    for epoch in range(1, max_epochs + 1):
        losses.append(log_loss(predict_proba(X, w, b), y))
        gw, gb = gradients(X, y, w, b)
        if np.sqrt(np.sum(gw ** 2) + gb ** 2) < grad_tol:
            converged = True
            break
        w -= lr * gw          # the update step: walk downhill
        b -= lr * gb
    seconds = time.perf_counter() - start
    return {"w": w, "b": b, "losses": losses, "epochs": len(losses),
            "converged": converged, "seconds": seconds}


def accuracy(X, y, w, b):
    return float(np.mean((predict_proba(X, w, b) >= 0.5) == y))


# ---------- reporting ----------

def plot_losses(results, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for lr, r in results.items():
        label = f"lr = {lr}" + ("" if r["converged"] else " (not converged)")
        ax.plot(range(1, r["epochs"] + 1), r["losses"], label=label, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("Training loss (cross-entropy)")
    ax.set_title("Training loss for three learning rates")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    features, X, y = load_data(path)
    X_tr_raw, X_te_raw, y_tr, y_te, _, test_idx = train_test_split(X, y)
    mean, std = fit_scaler(X_tr_raw)
    X_tr, X_te = scale(X_tr_raw, mean, std), scale(X_te_raw, mean, std)

    print(f"Loaded {len(y)} lunches, features: {', '.join(features)}")
    print(f"Split (seed {SEED}): {len(y_tr)} train, {len(y_te)} test")
    print(f"Good lunches: {y.mean():.0%} overall, {y_te.mean():.0%} in test")
    baseline = max(y_te.mean(), 1 - y_te.mean())
    print(f"Baseline (always guess the most common answer): {baseline:.1%} test accuracy")
    print(f"Convergence: gradient length < {GRAD_TOL}, max {MAX_EPOCHS} epochs\n")

    # --- three learning rates ---
    results = {}
    print(f"{'lr':>7} {'epochs':>7} {'converged':>10} {'time (s)':>9} "
          f"{'final loss':>11} {'train acc':>10} {'test acc':>9}")
    for lr in LEARNING_RATES:
        r = train(X_tr, y_tr, lr)
        results[lr] = r
        print(f"{lr:>7} {r['epochs']:>7} {str(r['converged']):>10} {r['seconds']:>9.3f} "
              f"{r['losses'][-1]:>11.5f} {accuracy(X_tr, y_tr, r['w'], r['b']):>10.1%} "
              f"{accuracy(X_te, y_te, r['w'], r['b']):>9.1%}")
    out = os.path.join(HERE, "loss_curves.png")
    plot_losses(results, out)
    print(f"\nSaved {out}")

    # --- the main model ---
    main_r = results[MAIN_LR]
    w, b = main_r["w"], main_r["b"]
    print(f"\n=== Main model (lr = {MAIN_LR}) ===")
    print(f"Test accuracy: {accuracy(X_te, y_te, w, b):.1%}")
    print(f"Converged in {main_r['epochs']} epochs, {main_r['seconds']:.3f} s")
    print("Weights (on scaled features; bigger size = stronger effect):")
    for name, wi in sorted(zip(features, w), key=lambda t: -abs(t[1])):
        print(f"  {name:<22} {wi:+.3f}")
    print(f"  {'bias':<22} {b:+.3f}")

    # --- error analysis ---
    p = predict_proba(X_te, w, b)
    wrong = np.where((p >= 0.5) != y_te)[0]
    print(f"\n=== Misclassified test lunches: {len(wrong)} of {len(y_te)} ===")
    print("Most confidently wrong first. p = predicted chance of a good lunch.")
    header = " ".join(f"{f[:12]:>12}" for f in features)
    print(f"  {'row':>4} {header} {'true':>5} {'p':>6}")
    for i in sorted(wrong, key=lambda i: -abs(p[i] - 0.5)):
        vals = " ".join(f"{v:>12.4g}" for v in X_te_raw[i])
        print(f"  {test_idx[i] + 2:>4} {vals} {int(y_te[i]):>5} {p[i]:>6.3f}")
    near = np.sum(np.abs(p[wrong] - 0.5) < 0.15)
    print(f"Wrong with p between 0.35 and 0.65 (near the boundary): {near} of {len(wrong)}")
    print("('row' is the line number in the CSV, counting the header as line 1.)")

    print("\nFeature averages by class (all 500 lunches):")
    for j, name in enumerate(features):
        print(f"  {name:<22} good: {X[y == 1, j].mean():>9.4g}   bad: {X[y == 0, j].mean():>9.4g}")

    # --- experiment: drop one feature at a time ---
    print(f"\n=== Drop one feature, retrain (lr = {MAIN_LR}) ===")
    print(f"  {'all features':<22} test acc {accuracy(X_te, y_te, w, b):.1%}")
    for j, name in enumerate(features):
        keep = [k for k in range(len(features)) if k != j]
        r = train(X_tr[:, keep], y_tr, MAIN_LR)
        acc = accuracy(X_te[:, keep], y_te, r["w"], r["b"])
        print(f"  without {name:<14} test acc {acc:.1%}")


if __name__ == "__main__":
    main()
