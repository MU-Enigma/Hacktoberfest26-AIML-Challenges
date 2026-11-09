import math


def sigmoid(x):
    """Stable sigmoid implementation."""
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)

    z = math.exp(x)
    return z / (1 + z)


def relu(x):
    """Return x if positive, otherwise return 0."""
    return max(0, x)


def tanh(x):
    """Squash a number into a value between -1 and 1."""
    return math.tanh(x)


def main():
    score = 47

    print("===== SQUASHING FUNCTION RESULTS =====")
    print(f"\nOriginal score: {score}")

    print("\n--- Sigmoid ---")
    print(f"Sigmoid({score}): {sigmoid(score):.6f}")

    print("\n--- ReLU ---")
    print(f"ReLU({score}): {relu(score):.6f}")

    print("\n--- Tanh ---")
    print(f"Tanh({score}): {tanh(score):.6f}")

    print("\n--- Additional Tests ---")
    for value in [-2, -1, 0, 1, 2]:
        print(
            f"x = {value:2}: "
            f"Sigmoid = {sigmoid(value):.4f}, "
            f"ReLU = {relu(value):.4f}, "
            f"Tanh = {tanh(value):.4f}"
        )

    print("\n--- Sigmoid Stability Test ---")
    try:
        naive_result = 1 / (1 + math.exp(1000))
        print("Naive sigmoid(-1000):", naive_result)
    except OverflowError:
        print("Naive sigmoid(-1000): OverflowError!")

    print("Stable sigmoid(-1000):", sigmoid(-1000))


if __name__ == "__main__":
    main()
