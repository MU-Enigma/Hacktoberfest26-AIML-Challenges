import math

def naivesigmoid(val):
    return 1.0 / (1.0 + math.exp(-val))

def sigmoid(val):
    if val >= 0.0:
        return 1.0 / (1.0 + math.exp(-val))
    ex = math.exp(val)
    return ex / (1.0 + ex)

def relu(val):
    if val > 0.0:
        return val
    return 0.0

def tanh(val):
    return math.tanh(val)

def run():
    print("testing naive sigmoid with extreme negative value:")
    brokeninput = -1000.0
    try:
        naivesigmoid(brokeninput)
    except OverflowError as err:
        print(f"naive sigmoid broke on {brokeninput}: {err}")
        print("reason: computing exp(1000) exceeds python float max limit (~1e308)")

    print("\ntesting fixed sigmoid:")
    print(f"sigmoid({brokeninput}) = {sigmoid(brokeninput)}")
    print(f"sigmoid(1000.0) = {sigmoid(1000.0)}")

    testvals = [-10.0, -2.0, 0.0, 2.0, 10.0, 47.0]
    print("\ncomparing sigmoid, relu, and tanh:")
    print("input\tsigmoid\trelu\ttanh")
    for num in testvals:
        sigout = sigmoid(num)
        reluout = relu(num)
        tanhout = tanh(num)
        print(f"{num}\t{sigout:.4f}\t{reluout:.4f}\t{tanhout:.4f}")

run()
