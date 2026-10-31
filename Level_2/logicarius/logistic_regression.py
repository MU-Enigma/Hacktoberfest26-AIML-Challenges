"""Level 2: logistic regression trained with gradient descent, pure Python.

Predicts good_lunch (1 or 0) from four numbers per lunch.
The gradient and the update step are written by hand (no scikit-learn).
"""
import csv
import math
import os
import random

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "binary_classification.csv")
FEATURES = ["queue_length", "plate_waste_fraction", "menu_board_kcal", "handwriting_neatness"]


# ---------- data ----------
def load_data(path):
    """Return a list of (features, label). features is a list of 4 floats."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            x = [float(row[name]) for name in FEATURES]
            y = int(row["good_lunch"])
            rows.append((x, y))
    return rows


def split(rows, seed=42, train_size=400):
    """Shuffle with a fixed seed, then cut into train and test."""
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    return rows[:train_size], rows[train_size:]


def fit_scaler(train_rows):
    """Min and max of each column, taken from TRAIN only (no peeking at test)."""
    lo, hi = [], []
    for j in range(4):
        column = [x[j] for x, y in train_rows]
        lo.append(min(column))
        hi.append(max(column))
    return lo, hi


def scale(rows, lo, hi):
    """Min-max normalization, same idea as Level 1: (x - lo) / (hi - lo)."""
    out = []
    for x, y in rows:
        out.append(([(x[j] - lo[j]) / (hi[j] - lo[j]) for j in range(4)], y))
    return out


# ---------- model ----------
def dot(a, b):
    total = 0.0
    for i in range(len(a)):
        total = total + a[i] * b[i]
    return total


def sigmoid(z):
    """Stable sigmoid (branch on the sign so exp never overflows)."""
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def predict_proba(w, b, x):
    """p = sigmoid(w . x + b): the probability that the lunch is good."""
    return sigmoid(dot(w, x) + b)


def loss(w, b, rows):
    """Average binary cross-entropy. eps keeps log() away from 0."""
    eps = 1e-12
    total = 0.0
    for x, y in rows:
        p = predict_proba(w, b, x)
        p = min(max(p, eps), 1 - eps)
        total = total + (-(y * math.log(p) + (1 - y) * math.log(1 - p)))
    return total / len(rows)


def accuracy(w, b, rows):
    correct = 0
    for x, y in rows:
        guess = 1 if predict_proba(w, b, x) >= 0.5 else 0
        if guess == y:
            correct = correct + 1
    return correct / len(rows)


# ---------- gradient descent ----------
def gradient_step(w, b, rows, lr):
    """One update, written by hand.

    For each row: error = p - y.
      dw[j] = average of error * x[j]
      db    = average of error
    Then move against the slope: w[j] = w[j] - lr * dw[j], b = b - lr * db.
    """
    n = len(rows)
    dw = [0.0] * len(w)
    db = 0.0
    for x, y in rows:
        error = predict_proba(w, b, x) - y
        for j in range(len(w)):
            dw[j] = dw[j] + error * x[j]
        db = db + error
    new_w = [w[j] - lr * dw[j] / n for j in range(len(w))]
    new_b = b - lr * db / n
    return new_w, new_b


def train(rows, lr, max_epochs=20000, tol=1e-6):
    """Run gradient descent. Stop when the loss improves by less than tol.

    Returns weights, bias, the loss after every epoch, and whether it converged.
    """
    w = [0.0] * len(rows[0][0])
    b = 0.0
    losses = [loss(w, b, rows)]
    converged = False
    for epoch in range(max_epochs):
        w, b = gradient_step(w, b, rows, lr)
        losses.append(loss(w, b, rows))
        if abs(losses[-2] - losses[-1]) < tol:
            converged = True
            break
    return w, b, losses, converged


# ---------- run ----------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_data(DATA)
    train_rows, test_rows = split(rows)
    lo, hi = fit_scaler(train_rows)
    train_s = scale(train_rows, lo, hi)
    test_s = scale(test_rows, lo, hi)

    plt.figure(figsize=(8, 5))
    results = {}
    for lr in [0.001, 0.1, 10]:
        w, b, losses, converged = train(train_s, lr)
        acc = accuracy(w, b, test_s)
        epochs = len(losses) - 1
        results[lr] = (w, b)
        print(f"lr={lr}: epochs={epochs} converged={converged} "
              f"final_train_loss={losses[-1]:.4f} test_accuracy={acc:.3f}")
        plt.plot(losses, label=f"lr = {lr}")

    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.title("Loss curves for three learning rates")
    plt.xscale("symlog")
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), "loss_curves.png"), dpi=150)

    # Weights of the best run (lr = 0.1), to see which features matter.
    w, b = results[0.1]
    print("weights at lr=0.1 (features scaled to 0..1):")
    for name, wj in zip(FEATURES, w):
        print(f"  {name}: {wj:.3f}")
