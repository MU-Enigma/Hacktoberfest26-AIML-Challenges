"""
squashing.py -- Level 1, Hacktoberfest "Similarity and Squashing".

"Squashing functions" take any real number and squeeze it into a fixed
range. These three show up everywhere in ML:

  * sigmoid: squashes to (0, 1)   -- read as "a probability"
  * ReLU:    squashes negatives to 0, leaves positives alone
  * tanh:    squashes to (-1, 1)  -- like sigmoid, but centered on 0

This file also demonstrates a real bug: a naive sigmoid implementation
overflows on large negative inputs, and shows the fix.

Pure Python only. `math` is allowed, NumPy is not.
"""

import math


def sigmoid(x):
    """Numerically stable sigmoid: 1 / (1 + e^-x), squashed into (0, 1).

    The naive formula 1 / (1 + math.exp(-x)) computes e^(-x). For a very
    negative x, -x is a very large POSITIVE number, and math.exp() of a
    large positive number overflows a 64-bit float (OverflowError) around
    x <= -710 or so (e^710 is already bigger than the biggest float).

    The fix: branch on the sign of x.
      - If x >= 0, -x <= 0, so e^-x is between 0 and 1 -- always safe.
        Use the original formula: 1 / (1 + e^-x).
      - If x < 0, flip the formula algebraically so the exponent is of a
        NEGATIVE number instead (e^x, which just shrinks towards 0, never
        overflows):
            1 / (1 + e^-x)
          = e^x / (e^x (1 + e^-x))      [multiply top and bottom by e^x]
          = e^x / (e^x + e^x * e^-x)
          = e^x / (e^x + 1)             [e^x * e^-x = e^0 = 1]
        So for x < 0 we compute e^x / (1 + e^x) instead -- mathematically
        the exact same function, just an expression that never needs a
        huge exponent.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def sigmoid_naive(x):
    """The textbook formula, with no overflow protection.

    Kept around ONLY to demonstrate the bug in main() below -- do not use
    this version for real work, use sigmoid() instead.
    """
    return 1.0 / (1.0 + math.exp(-x))


def relu(x):
    """Rectified Linear Unit: x if x > 0, else 0.

    The simplest squashing function here -- it doesn't bound the top, it
    just clips off everything below zero.
    """
    return x if x > 0 else 0.0


def tanh(x):
    """Hyperbolic tangent, squashed into (-1, 1).

    math.tanh is part of the standard `math` module (not NumPy), so this
    is allowed by the challenge rules. We still wrap it in our own function
    so every squashing function in this file has the same shape and a
    docstring that explains what it does.
    """
    return math.tanh(x)


def demo_naive_sigmoid_overflow():
    """Show the naive sigmoid breaking on a large negative input, then show
    the stable version handling the same input fine.
    """
    print("--- Naive sigmoid on a large negative input ---")
    x = -1000.0
    print(f"sigmoid_naive({x}) ...")
    try:
        result = sigmoid_naive(x)
        print(f"  -> {result}  (no crash this time)")
    except OverflowError as e:
        print(f"  -> OverflowError: {e}")
        print("  Why: math.exp(-x) = math.exp(1000.0), and e^1000 is far")
        print("  bigger than the largest float Python can represent")
        print("  (roughly e^709.78 before overflow). The naive formula")
        print("  needs that huge number to exist as an intermediate value,")
        print("  even though the final answer is just a tiny number near 0.")

    print("\n--- Stable sigmoid on the same input ---")
    result = sigmoid(x)
    print(f"sigmoid({x}) -> {result}")
    print("  Why it works: for x < 0 we compute e^x / (1 + e^x) instead.")
    print("  e^x for a very negative x is e^(-1000), i.e. a number very")
    print("  close to 0 -- easy for a float to hold. Same math, different")
    print("  (safer) arithmetic path.")

    # Also show the boundary the hint in the spec points at: math.exp()
    # itself overflows for an argument a bit beyond ~709.78, so a naive
    # sigmoid breaks once -x crosses that line, i.e. x below roughly -710.
    print("\n--- Where exactly math.exp() itself overflows ---")
    for x in (-700.0, -709.0, -710.0, -800.0):
        try:
            math.exp(-x)
            print(f"  math.exp(-({x})) = math.exp({-x}) -> ok")
        except OverflowError:
            print(f"  math.exp(-({x})) = math.exp({-x}) -> OverflowError")


def demo_all_functions():
    """Print sigmoid/relu/tanh on a handful of sample inputs, so the output
    of this script alone shows all three functions behaving sensibly.
    """
    print("\n--- sigmoid / relu / tanh on a few sample inputs ---")
    samples = [-10.0, -1.0, 0.0, 0.5, 1.0, 10.0]
    for x in samples:
        print(
            f"x={x:>6}  sigmoid={sigmoid(x):.6f}  "
            f"relu={relu(x):.6f}  tanh={tanh(x):.6f}"
        )


def main():
    demo_naive_sigmoid_overflow()
    demo_all_functions()


if __name__ == "__main__":
    main()
