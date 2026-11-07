import math

# naive implementation
# can trigger computer arithmetic overflow errors when exp(-x) 
# exceeds the maximum representable floating-point number
def naive_sigmoid(x):
    return 1 / (1 + math.exp(-x))

# numerically stable implementation
def sigmoid(x):

    if x >= 0:
        return 1 / (1 + math.exp(-x))

    exp_x = math.exp(x)

    return exp_x / (1 + exp_x)


# naive implementation
def naive_relu(x):
    return max(0, x)

# improved implementation
def relu(x):
    return x if x > 0 else 0

# naive implementation 
# can trigger computer arithmetic overflow errors when exp(2*x) 
# exceeds the maximum representable floating-point number
def naive_tanh(x):
    return (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)

# numerically stable implementation
def tanh(x):

    if x == 0:
        return 0.0

    exp_term = math.exp(-2 * abs(x))
    result = (1 - exp_term) / (1 + exp_term)

    return math.copysign(result, x)

# testing 
def test_squashing_functions():

    print("\nSquashing functions tests")

    test_values = [-1000, -2, -1, 0, 1, 2, 1000]

    print(
        f"\n{'Input':>10} {'Sigmoid':>15} "
        f"{'ReLU':>15} {'Tanh':>15}"
    )

    for x in test_values:

        print(
            f"{x:10} "
            f"{sigmoid(x):15.6f} "
            f"{relu(x):15.6f} "
            f"{tanh(x):15.6f}"
        )

# testing with extreme values to check for numerical stability
def test_overflow():

    print("\nNumerical stability tests")

    # Sigmoid: large negative input
    print("\n1. Sigmoid")

    try:
        print("Naive sigmoid(-1000):", naive_sigmoid(-1000))
    except OverflowError as error:
        print("Naive sigmoid failed:", error)

    print("Stable sigmoid(-1000):", sigmoid(-1000))


    # Tanh: large positive input
    print("\n2. TanH")

    try:
        print("Naive tanh(1000):", naive_tanh(1000))
    except OverflowError as error:
        print("Naive tanh failed:", error)

    print("Stable tanh(1000):", tanh(1000))


    # ReLU: extreme inputs
    print("\n3. ReLU")

    for x in [-1000, 0, 1000]:
        print(
            f"Input: {x:5} | "
            f"Naive: {naive_relu(x):10} | "
            f"Improved: {relu(x):10}"
        )

if __name__ == "__main__":

    test_overflow()
    test_squashing_functions()
    