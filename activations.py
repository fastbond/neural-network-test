import numpy as np

def relu(x):
    return np.maximum(0,x)

def relu_d(x):
    return np.where(x <= 0, 0, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))
    # Equivalent to?
    #return np.exp(-x) / np.power(1 + np.exp(-x), 2)

def tanh(x):
    return np.tanh(x)

def tanh_d(x):
    return 1 - np.power(np.tanh(x),2)

# Temperature controls "confidence"
# AKA low temperature(<1) means high values will be counted more strongly, small values even smaller
# High temperature(>1) means everything is more similar
def softmax(X, temp=1):
    E = np.exp(X)
    sum = np.sum(np.exp(X))
    return E / sum

def softmax_d(X):
    return softmax(X)
    #raise NotImplementedError