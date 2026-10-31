import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. Core Mathematical Functions
# ==========================================
def sigmoid(z):
    # Clip values to prevent overflow errors in np.exp
    z = np.clip(z, -250, 250)
    return 1.0 / (1.0 + np.exp(-z))

def compute_loss(y_true, y_pred):
    # Add a small epsilon to prevent log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# ==========================================
# 2. Logistic Regression & Gradient Descent
# ==========================================
def train_logistic_regression(X, y, lr, max_iters=10000, tol=1e-6):
    """
    Trains a logistic regression model using gradient descent.
    Convergence criterion: Stops when the change in loss is less than `tol`.
    """
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0.0
    losses = []

    for i in range(max_iters):
        # Forward pass
        linear_model = np.dot(X, weights) + bias
        predictions = sigmoid(linear_model)

        # Compute and store loss
        loss = compute_loss(y, predictions)
        losses.append(loss)

        # Convergence Check
        if i > 0 and abs(losses[-2] - losses[-1]) < tol:
            print(f"Converged at iteration {i} for LR={lr}")
            break

        # Compute Gradients
        dw = (1 / m) * np.dot(X.T, (predictions - y))
        db = (1 / m) * np.sum(predictions - y)

        # Update weights and bias
        weights -= lr * dw
        bias -= lr * db

    return weights, bias, losses

def predict(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    return np.array([1 if i > 0.5 else 0 for i in y_pred])

def calculate_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# ==========================================
# 3. Data Loading and Preprocessing
# ==========================================
def main():
    # Load dataset (adjust target column name if it is not the last column)
    df = pd.read_csv('../../datasets/binary_classification.csv')
    
    X_raw = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Normalize features (Z-score normalization)
    # Critical for gradient descent to converge properly
    mean = np.mean(X_raw, axis=0)
    std = np.std(X_raw, axis=0)
    X = (X_raw - mean) / (std + 1e-8)

    # Manual Train/Test Split (80/20) with fixed random seed
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    train_size = int(0.8 * len(X))
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ==========================================
    # 4. Training with Multiple Learning Rates
    # ==========================================
    learning_rates = [0.1, 0.01, 0.001]
    results = {}

    plt.figure(figsize=(10, 6))

    for lr in learning_rates:
        print(f"--- Training with Learning Rate: {lr} ---")
        weights, bias, losses = train_logistic_regression(X_train, y_train, lr=lr)
        results[lr] = {'weights': weights, 'bias': bias}
        
        # Plot loss curve
        plt.plot(losses, label=f'LR = {lr}')
        
        # Test Set Evaluation
        preds = predict(X_test, weights, bias)
        acc = calculate_accuracy(y_test, preds)
        print(f"Test Accuracy (LR={lr}): {acc:.4f}\n")

    # Format and save plot
    plt.title('Loss Curves for Different Learning Rates')
    plt.xlabel('Iterations')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('loss_curves.png')
    print("Saved 'loss_curves.png'")

if __name__ == "__main__":
    main()
