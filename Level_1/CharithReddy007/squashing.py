"""Level 1, Task 2: squashing functions (sigmoid, ReLU, tanh).

Pure Python only (math is allowed, NumPy is not).

Run:  python squashing.py
"""

import math


def naive_sigmoid(x):
    """Textbook formula 1 / (1 + e^-x). Breaks for large negative x."""
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid(x):
    """Numerically stable sigmoid.

    For x >= 0, e^-x is between 0 and 1, so the textbook formula is safe.
    For x < 0, multiply top and bottom by e^x to get e^x / (1 + e^x).
    Now we only ever compute e^(negative number), which can shrink toward 0
    but can never overflow.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def relu(x):
    """Keep positives, turn negatives into 0."""
    return x if x > 0 else 0.0


def tanh(x):
    """Stable tanh, built from exponentials (no math.tanh).

    tanh(x) = (1 - e^-2x) / (1 + e^-2x) for x >= 0, and tanh is odd
    (tanh(-x) = -tanh(x)), so we only ever compute e^(negative number).
    """
    if x < 0:
        return -tanh(-x)
    t = math.exp(-2.0 * x)
    return (1.0 - t) / (1.0 + t)


def demo():
    print("Breaking the naive sigmoid:")
    for x in [-10, -100, -700, -710, -1000]:
        try:
            print(f"  naive_sigmoid({x}) = {naive_sigmoid(x)}")
        except OverflowError as e:
            print(f"  naive_sigmoid({x}) -> OverflowError: {e}")

    print("\nFixed sigmoid:")
    for x in [-1000, -710, -10, 0, 10, 710, 1000]:
        print(f"  sigmoid({x}) = {sigmoid(x)}")

    print("\nReLU and tanh:")
    for x in [-1000, -2, -0.5, 0, 0.5, 2, 1000]:
        print(f"  x={x:>6}: relu={relu(x):>7}  tanh={tanh(x): .6f}  "
              f"(math.tanh={math.tanh(x): .6f})")


if __name__ == "__main__":
    demo()
