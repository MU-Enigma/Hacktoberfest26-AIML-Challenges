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
