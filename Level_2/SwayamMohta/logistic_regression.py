"""Level 2: Logistic Regression with Gradient Descent.

Implemented from scratch using NumPy and Pandas (NO scikit-learn).
Predicts whether mess lunch is good (good_lunch = 1) from 4 numeric features:
- queue_length
- plate_waste_fraction
- menu_board_kcal
- handwriting_neatness

Trains with multiple learning rates, plots loss curves, reports test accuracy
and convergence times.
"""

import os
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function for NumPy arrays.

    Uses negative branching to prevent overflow in exp(-z).
    """
    z = np.asarray(z, dtype=np.float64)
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


def compute_bce_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Binary Cross-Entropy (Log Loss).

    L = -(1/m) * sum(y * log(y_hat) + (1-y) * log(1 - y_hat))
    """
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    return float(
        -np.mean(
            y_true * np.log(y_pred_clipped)
            + (1.0 - y_true) * np.log(1.0 - y_pred_clipped)
        )
    )


class LogisticRegressionScratch:
    """Logistic Regression trained via Batch Gradient Descent."""

    def __init__(self, learning_rate: float = 0.1, max_iters: int = 2000, tol: float = 1e-6):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.weights = None
        self.bias = 0.0
        self.loss_history: List[float] = []
        self.converged_iter = max_iters
        self.fit_time_sec = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        """Train weights and bias using gradient descent on binary cross entropy."""
        m, n = X.shape
        self.weights = np.zeros(n, dtype=np.float64)
        self.bias = 0.0
        self.loss_history = []

        start_time = time.time()
        for i in range(self.max_iters):
            # 1. Forward pass
            linear_output = np.dot(X, self.weights) + self.bias
            predictions = stable_sigmoid(linear_output)

            # 2. Compute loss
            loss = compute_bce_loss(y, predictions)
            self.loss_history.append(loss)

            # 3. Check convergence criterion (|L_t - L_{t-1}| < tol)
            if i > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                self.converged_iter = i
                break

            # 4. Backward pass (Gradients of BCE loss wrt w and b)
            errors = predictions - y
            dw = (1.0 / m) * np.dot(X.T, errors)
            db = (1.0 / m) * np.sum(errors)

            # 5. Parameter updates
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        self.fit_time_sec = time.time() - start_time
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities for positive class."""
        linear_output = np.dot(X, self.weights) + self.bias
        return stable_sigmoid(linear_output)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return discrete binary predictions (0 or 1)."""
        return (self.predict_proba(X) >= threshold).astype(int)


def prepare_data(
    csv_path: str,
    feature_cols: List[str],
    target_col: str,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series, pd.DataFrame]:
    """Load dataset, perform train/test split with fixed seed, and standardize features."""
    df = pd.read_csv(csv_path)

    # Deterministic train/test split
    np.random.seed(seed)
    shuffled_idx = np.random.permutation(len(df))
    split_point = int((1.0 - test_ratio) * len(df))
    train_idx = shuffled_idx[:split_point]
    test_idx = shuffled_idx[split_point:]

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Compute standardization statistics strictly from the training set
    feature_means = train_df[feature_cols].mean()
    feature_stds = train_df[feature_cols].std()

    X_train = ((train_df[feature_cols] - feature_means) / feature_stds).values
    y_train = train_df[target_col].values

    X_test = ((test_df[feature_cols] - feature_means) / feature_stds).values
    y_test = test_df[target_col].values

    return X_train, y_train, X_test, y_test, feature_means, feature_stds, test_df


def plot_learning_rates(
    results: Dict[float, List[float]],
    save_path: str,
) -> None:
    """Plot and save training loss curves for multiple learning rates."""
    plt.figure(figsize=(10, 6))

    colors = {0.01: "#e67e22", 0.1: "#2980b9", 1.0: "#27ae60"}
    for lr, losses in results.items():
        color = colors.get(lr, None)
        label = f"lr = {lr}"
        plt.plot(losses, label=label, color=color, linewidth=2)

    plt.title("Level 2: Binary Cross-Entropy Loss Curves Across Learning Rates", fontsize=14, fontweight="bold")
    plt.xlabel("Iteration (Epoch)", fontsize=12)
    plt.ylabel("Training Loss (Binary Cross-Entropy)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved loss curve plot to {save_path}")


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.normpath(
        os.path.join(base_dir, "..", "..", "datasets", "binary_classification.csv")
    )
    feature_cols = [
        "queue_length",
        "plate_waste_fraction",
        "menu_board_kcal",
        "handwriting_neatness",
    ]
    target_col = "good_lunch"

    print("=" * 70)
    print(" LEVEL 2: LOGISTIC REGRESSION WITH GRADIENT DESCENT")
    print("=" * 70)

    # 1. Prepare Data
    X_train, y_train, X_test, y_test, means, stds, test_df = prepare_data(
        dataset_path, feature_cols, target_col, test_ratio=0.2, seed=42
    )
    print(f"Dataset split: {len(X_train)} training samples, {len(X_test)} test samples (seed=42).")

    # 2. Train with Three Learning Rates
    # Predictions before running:
    # - lr = 0.01: Very slow steady convergence, will likely hit max iterations without fully converging.
    # - lr = 0.1:  Optimal learning rate; smooth steady descent, converges in a few hundred iterations.
    # - lr = 1.0:  Aggressive descent; rapid early drop, converges in under 100 iterations.
    learning_rates = [0.01, 0.1, 1.0]
    loss_histories = {}
    models = {}

    print("\n" + "-" * 70)
    print("TRAINING COMPARISON ACROSS LEARNING RATES:")
    print("-" * 70)
    print("Convergence Criterion: Change in training loss between consecutive iterations < 1e-6\n")

    for lr in learning_rates:
        model = LogisticRegressionScratch(learning_rate=lr, max_iters=2000, tol=1e-6)
        model.fit(X_train, y_train)

        test_preds = model.predict(X_test)
        test_acc = np.mean(test_preds == y_test)
        test_loss = compute_bce_loss(y_test, model.predict_proba(X_test))

        loss_histories[lr] = model.loss_history
        models[lr] = model

        print(f"Learning Rate: {lr}")
        print(f"  - Converged Iterations : {model.converged_iter} / {model.max_iters}")
        print(f"  - Training Time        : {model.fit_time_sec * 1000:.2f} ms")
        print(f"  - Final Training Loss  : {model.loss_history[-1]:.6f}")
        print(f"  - Test Accuracy        : {test_acc * 100:.2f}%")
        print(f"  - Test BCE Loss        : {test_loss:.6f}\n")

    # 3. Save Loss Curves Plot
    plot_path = os.path.join(base_dir, "loss_curves.png")
    plot_learning_rates(loss_histories, plot_path)

    # 4. Error Analysis Details
    best_model = models[0.1]
    test_probs = best_model.predict_proba(X_test)
    test_predictions = (test_probs >= 0.5).astype(int)

    analysis_df = test_df.copy()
    analysis_df["predicted"] = test_predictions
    analysis_df["probability"] = test_probs
    analysis_df["is_error"] = analysis_df[target_col] != analysis_df["predicted"]

    errors = analysis_df[analysis_df["is_error"]]
    print("-" * 70)
    print(f"ERROR ANALYSIS (Model gets {len(errors)} / {len(test_df)} test samples wrong):")
    print("-" * 70)
    print("Learned Weights:")
    for f, w in zip(feature_cols, best_model.weights):
        print(f"  - {f:<22}: {w:+.4f}")
    print(f"  - bias                  : {best_model.bias:+.4f}")

    print("\nSample Misclassified Examples:")
    for idx, row in errors.head(3).iterrows():
        print(
            f"  Index {idx:3d} | Actual={int(row[target_col])} Pred={int(row['predicted'])} "
            f"(P={row['probability']:.3f}) | queue={row['queue_length']}, "
            f"waste={row['plate_waste_fraction']}, kcal={row['menu_board_kcal']}, "
            f"neatness={row['handwriting_neatness']}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
