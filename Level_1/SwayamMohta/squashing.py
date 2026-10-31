"""Squashing Functions: Sigmoid, ReLU, and Tanh.

Pure Python implementation (standard library only, NO NumPy).
Includes demonstration of numerical overflow in naive sigmoid for large negative inputs
and its robust, numerically stable resolution.
"""

import math
from typing import Union

Number = Union[int, float]


def relu(x: Number) -> float:
    """Rectified Linear Unit: max(0, x)."""
    return float(max(0.0, float(x)))


def tanh(x: Number) -> float:
    """Hyperbolic tangent function: (e^x - e^-x) / (e^x + e^-x).

    Squashes real numbers into (-1, 1).
    Utilizes math.tanh for numerical stability across extreme values.
    """
    return math.tanh(float(x))


def sigmoid_naive(x: Number) -> float:
    """Naive implementation of sigmoid: 1 / (1 + e^(-x)).

    Vulnerable to OverflowError when x is a large negative number
    because e^(-x) exceeds the maximum representable float.
    """
    return 1.0 / (1.0 + math.exp(-float(x)))


def sigmoid(x: Number) -> float:
    """Numerically stable sigmoid function.

    Formula:
      For x >= 0: 1 / (1 + e^(-x))
      For x < 0:  e^x / (1 + e^x)

    When x is a large negative number (e.g. x = -1000), e^x approaches 0.0,
    cleanly avoiding OverflowError and returning 0.0.
    """
    val = float(x)
    if val >= 0:
        return 1.0 / (1.0 + math.exp(-val))
    else:
        exp_val = math.exp(val)
        return exp_val / (1.0 + exp_val)


def demonstrate_squashing() -> None:
    print("=" * 70)
    print(" LEVEL 1: SQUASHING FUNCTIONS (SIGMOID, RELU, TANH)")
    print("=" * 70)

    test_values = [-10.0, -2.0, -0.5, 0.0, 0.5, 2.0, 10.0]
    print(f"\n{'x':>8} | {'Sigmoid':>12} | {'ReLU':>10} | {'Tanh':>12}")
    print("-" * 50)
    for x in test_values:
        print(f"{x:8.2f} | {sigmoid(x):12.6f} | {relu(x):10.4f} | {tanh(x):12.6f}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION OF NAIVE SIGMOID OVERFLOW & FIX:")
    print("=" * 70)
    large_neg = -1000.0
    print(f"Testing input x = {large_neg}:")

    # 1. Naive attempt
    try:
        val = sigmoid_naive(large_neg)
        print(f"  Naive sigmoid({large_neg}) = {val}")
    except OverflowError as err:
        print(f"  [ERROR] Naive sigmoid({large_neg}) crashed with: {err}")
        print("  Explanation: In 1 / (1 + e^(-x)), when x = -1000, e^(1000) exceeds")
        print("  the IEEE 754 float maximum (~1.79e+308), causing Python's math.exp")
        print("  to throw an OverflowError.")

    # 2. Robust fix
    robust_val = sigmoid(large_neg)
    print(f"\n  [FIXED] Robust sigmoid({large_neg}) = {robust_val}")
    print("  Explanation: By algebraically rewriting 1 / (1 + e^(-x)) as e^x / (1 + e^x)")
    print("  for x < 0, e^(-1000) simply underflows to 0.0, safely yielding 0.0 / (1.0 + 0.0) = 0.0.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_squashing()
