"""Squashing Functions (Level 1).

This script implements:
- Naive sigmoid (and shows where and why it breaks)
- Numerically stable sigmoid (the fix)
- ReLU (Rectified Linear Unit)
- Tanh (Hyperbolic Tangent, scratch stable formulation and standard math.tanh)

Pure Python: only standard library (math) is used, no NumPy.
"""

import math


def naive_sigmoid(x):
    """Naive sigmoid implementation: 1 / (1 + exp(-x)).

    Susceptible to OverflowError when x is a large negative number,
    because exp(-x) becomes exp(|huge|) which exceeds float capacity.
    """
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid(x):
    """Numerically stable sigmoid implementation.

    Mathematical identity:
      For x >= 0: 1 / (1 + exp(-x))
      For x <  0: exp(x) / (1 + exp(x))

    Why this works:
      When x >= 0, -x <= 0, so exp(-x) is in (0, 1] and never overflows.
      When x < 0,  exp(x) is in (0, 1) and never overflows (as x -> -inf, exp(x) -> 0.0).
    """
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def relu(x):
    """Rectified Linear Unit (ReLU): returns max(0, x)."""
    return max(0.0, float(x))


def tanh(x):
    """Numerically stable hyperbolic tangent function.

    Formula:
      tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Stable branching:
      For x >= 0: (1 - exp(-2x)) / (1 + exp(-2x))
      For x <  0: (exp(2x) - 1) / (exp(2x) + 1)
    """
    if x >= 0:
        z = math.exp(-2.0 * x)
        return (1.0 - z) / (1.0 + z)
    else:
        z = math.exp(2.0 * x)
        return (z - 1.0) / (z + 1.0)


def demonstrate_break_and_fix():
    """Demonstrates how naive_sigmoid breaks on large negative inputs and how stable sigmoid fixes it."""
    breaking_input = -1000

    print("=" * 70)
    print(" LEVEL 1: SQUASHING FUNCTIONS (SIGMOID BREAK & FIX)")
    print("=" * 70)
    print(f"Testing naive_sigmoid with large negative input: x = {breaking_input}")

    try:
        val = naive_sigmoid(breaking_input)
        print(f"naive_sigmoid({breaking_input}) = {val}")
    except OverflowError as err:
        print(f"[CAUGHT EXPECTED ERROR] naive_sigmoid({breaking_input}) raised: {type(err).__name__}: {err}")
        print("\n--- WHY DID IT BREAK? ---")
        print(
            f"In naive_sigmoid(x) = 1 / (1 + exp(-x)), when x = {breaking_input},\n"
            f"-x becomes +{abs(breaking_input)}.\n"
            f"Standard 64-bit IEEE 754 floating point numbers can only represent values up to ~1.79e+308.\n"
            f"Because exp(709.78) ~ 1.79e+308, calculating exp({abs(breaking_input)}) attempts to evaluate\n"
            f"e^{abs(breaking_input)}, which astronomically exceeds the maximum float limit, causing an OverflowError.\n"
        )

    print("--- TESTING THE STABLE FIX ---")
    fixed_val = sigmoid(breaking_input)
    print(f"stable sigmoid({breaking_input}) = {fixed_val}")
    print(
        "Explanation of fix:\n"
        "By rewriting 1 / (1 + exp(-x)) as exp(x) / (1 + exp(x)) whenever x < 0,\n"
        f"we compute exp({breaking_input}) instead of exp(+{abs(breaking_input)}).\n"
        f"exp({breaking_input}) smoothly underflows to 0.0, yielding exactly 0.0 / (1.0 + 0.0) = 0.0.\n"
    )


def test_suite():
    """Displays output of sigmoid, ReLU, and tanh across representative inputs."""
    sample_inputs = [-1000.0, -100.0, -10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0, 100.0, 1000.0]

    print("=" * 70)
    print(" SQUASHING FUNCTION OUTPUT TABLE")
    print("=" * 70)
    print(f"{'Input (x)':<12}{'Sigmoid(x)':<15}{'ReLU(x)':<15}{'Tanh(x)':<15}{'math.tanh(x)':<15}")
    print("-" * 72)
    for x in sample_inputs:
        s = sigmoid(x)
        r = relu(x)
        t = tanh(x)
        mt = math.tanh(x)
        print(f"{x:<12.1f}{s:<15.6f}{r:<15.6f}{t:<15.6f}{mt:<15.6f}")
    print("=" * 70)


def main():
    demonstrate_break_and_fix()
    test_suite()


if __name__ == "__main__":
    main()

