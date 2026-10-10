import math

def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)

def relu(z):
    return max(0.0, float(z))

def tanh(z):
    if z >= 0:
        e_minus_2z = math.exp(-2 * z)
        return (1 - e_minus_2z) / (1 + e_minus_2z)
    
    e_2z = math.exp(2 * z)
    return (e_2z - 1) / (e_2z + 1)

# These have definite naive forms but that just causes an OverFlowError because 
# how can your pookie level brained computer compute e power 1000. 
# My dumb ahh brain couldnt even compute e power 2.
#Yea thats it... squashing functions. Yay. 


# Gosh is this why people didnt choose to pr this..