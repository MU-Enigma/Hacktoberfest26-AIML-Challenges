
import math


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        value = math.exp(x)
        return value / (1 + value)


def relu(x):
    return max(0, x)


def tanh(x):
    return math.tanh(x)


numbers = [-2, -1, 0, 1, 2]

print("Input\tSigmoid\tReLU\tTanh")

for x in numbers:
    print(
        x, "\t",
        round(sigmoid(x), 4), "\t",
        round(relu(x), 4), "\t",
        round(tanh(x), 4)
    )

print("\nTesting sigmoid with -1000:")

try:
    print(1 / (1 + math.exp(1000)))
except OverflowError:
    print("OverflowError: exponential is too large.")

print("Fixed sigmoid(-1000):", sigmoid(-1000))
