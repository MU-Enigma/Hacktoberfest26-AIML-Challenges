import math

def sigmoid_naive(x):
    return 1/(1 + math.exp(-x));
def sigmoid(x):
    if x>=0: 
        return 1/(1 + math.exp(-x));
    else: 
        e = math.exp(x)
        return e/(1+e)
print(sigmoid(0))
print(sigmoid(-710))
print(sigmoid(-1000))
print(sigmoid(1000))