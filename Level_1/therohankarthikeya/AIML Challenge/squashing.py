import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0.0, float(x))

def tanh(x):
    return math.tanh(x)

#Yea thats it... squashing functions. Yay. Gosh is this why people didnt choose to pr this..