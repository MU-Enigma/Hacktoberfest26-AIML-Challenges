import math


def sigmoid_naive(x):
    return 1 / (1 + math.exp(-x))


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def relu(x):
    return max(0.0, x)


def tanh(x):
    return 2 * sigmoid(2 * x) - 1


if __name__ == "__main__":
    print("x      sigmoid   relu   tanh(mine)  tanh(math)")
    for x in (-5, -1, 0, 1, 5):
        print(f"{x:<6} {sigmoid(x):.4f}    {relu(x):<6} "
              f"{tanh(x):.4f}      {math.tanh(x):.4f}")

    try:
        print(sigmoid_naive(-1000))
    except OverflowError as e:
        print("naive sigmoid broke at -1000:", e)
    print("fixed sigmoid at -1000:", sigmoid(-1000))
