
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "datasets" / "binary_classification.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "queue_length",
    "plate_waste_fraction",
    "menu_board_kcal",
    "handwriting_neatness",
]
TARGET = "good_lunch"

SEED = 42
EPOCHS = 10000
TOLERANCE = 1e-9
PATIENCE = 20


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y, probabilities):
    eps = 1e-12
    probabilities = np.clip(probabilities, eps, 1 - eps)
    return -np.mean(
        y * np.log(probabilities)
        + (1 - y) * np.log(1 - probabilities)
    )


def train_model(X, y, learning_rate):
    """Train logistic regression using gradient descent."""
    weights = np.zeros(X.shape[1])
    bias = 0.0
    losses = []
    stable_epochs = 0
    converged = False
    start = time.perf_counter()

    for epoch in range(EPOCHS):
        probabilities = sigmoid(X @ weights + bias)
        loss = binary_cross_entropy(y, probabilities)
        losses.append(loss)

        # Calculate gradients manually.
        error = probabilities - y
        dw = (X.T @ error) / len(y)
        db = np.mean(error)

        # Update model parameters.
        weights -= learning_rate * dw
        bias -= learning_rate * db

        if len(losses) > 1:
            if abs(losses[-2] - losses[-1]) < TOLERANCE:
                stable_epochs += 1
            else:
                stable_epochs = 0

        if stable_epochs >= PATIENCE:
            converged = True
            break

    elapsed = time.perf_counter() - start
    return weights, bias, losses, elapsed, epoch + 1, converged


def predict(X, weights, bias):
    return (sigmoid(X @ weights + bias) >= 0.5).astype(int)


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    # Reproducible 80/20 train-test split.
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(X))
    split = int(0.8 * len(X))

    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Standardize using training data only.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    learning_rates = [0.001, 0.05, 0.5]
    results = {}
    print(f"Dataset: {len(df)} rows")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print("\nMODEL RESULTS")

    for lr in learning_rates:
        weights, bias, losses, elapsed, epochs, converged = train_model(
            X_train, y_train, lr
        )

        predictions = predict(X_test, weights, bias)
        accuracy = np.mean(predictions == y_test) * 100

        results[lr] = (weights, bias, losses, predictions)

        print(f"\nLearning rate: {lr}")
        print(f"Test accuracy: {accuracy:.2f}%")
        print(f"Epochs: {epochs}")
        print(f"Converged: {'Yes' if converged else 'No (maximum epochs reached)'}")
        print(f"Convergence time: {elapsed:.6f} seconds")
        print(f"Final loss: {losses[-1]:.6f}")

    # Plot all three loss curves.
    plt.figure(figsize=(9, 5))
    for lr, (_, _, losses, _) in results.items():
        plt.plot(losses, label=f"Learning rate = {lr}")

    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title("Logistic Regression: Learning Rate Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curves.png")
    plt.close()

    # Use the middle learning rate for error analysis.
    lr = 0.05
    weights, bias, losses, predictions = results[lr]
    wrong = np.where(predictions != y_test)[0]

    print("\nTHREE INCORRECT PREDICTIONS")
    if len(wrong) == 0:
        print("No errors on this test split.")
    else:
        selected = wrong[:3]
        error_table = pd.DataFrame(
            X_test[selected] * std + mean,
            columns=FEATURES
        )
        error_table["actual"] = y_test[selected].astype(int)
        error_table["predicted"] = predictions[selected]
        print(error_table.to_string(index=False))
        error_table.to_csv(
            OUTPUT_DIR / "misclassified_examples.csv",
            index=False
        )

    # Hypothesis test: remove handwriting_neatness.
    reduced_features = [i for i, name in enumerate(FEATURES)
                        if name != "handwriting_neatness"]

    Xtr_reduced = X_train[:, reduced_features]
    Xte_reduced = X_test[:, reduced_features]

    w, b, _, _, _, _ = train_model(Xtr_reduced, y_train, lr)
    reduced_predictions = predict(Xte_reduced, w, b)
    reduced_accuracy = np.mean(reduced_predictions == y_test) * 100

    baseline_accuracy = np.mean(predictions == y_test) * 100

    print("\nHYPOTHESIS TEST: REMOVE HANDWRITING NEATNESS")
    print(f"Original accuracy: {baseline_accuracy:.2f}%")
    print(f"Reduced accuracy: {reduced_accuracy:.2f}%")
    print(f"Change: {reduced_accuracy - baseline_accuracy:+.2f} percentage points")

    print("\nLoss graph saved to:", OUTPUT_DIR / "loss_curves.png")
    print("Error examples saved to:", OUTPUT_DIR / "misclassified_examples.csv")


if __name__ == "__main__":
    main()
