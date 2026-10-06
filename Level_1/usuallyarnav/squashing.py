import math

def sigmoid_naive(x):
    return 1/(1 + math.exp(-x));
def sigmoid(x):
    if x>=0: 
        return 1/(1 + math.exp(-x));
    else: 
        e = math.exp(x)
        return e/(1+e)

# there we go using tanh to squash 
def tanh(x):
    if x < 0:
        return -tanh(-x)

    t = math.exp(-2 * x)
    return (1 - t) / (1 + t)

# now the last one relu the rectiflied linear unit

def relu(x):
    if math.isnan(x):
        return x

    return x if x > 0 else 0.0

if __name__ == "__main__":
    try:
        sigmoid_naive(-710)
    except OverflowError as err:
        print("sigmoid_naive(-710) crashed:", err)
    print("sigmoid(-710) =", sigmoid(-710))
    print("tanh(1000) =", tanh(1000), "| tanh(-1000) =", tanh(-1000))
    print("relu(-3) =", relu(-3), "| relu(2.5) =", relu(2.5))
    