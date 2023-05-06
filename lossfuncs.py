from abc import ABC, abstractmethod
import numpy as np

# TODO
class LossFunction(ABC):
    def __init__(self):
        self.func = None
        self.deriv = None

# Sum of squared errors for all samples divided by num samples
# Will this work for batch?
# https://stackoverflow.com/questions/55936214/correct-way-to-calculate-mse-for-autoencoders-with-batch-training
def mse(y_pred, y_true):
    #return np.mean(np.power(y_true - y_pred, 2))
    return np.mean(np.mean(np.power(y_true - y_pred, 2), axis=1))

def mse_d(y_pred, y_true):
    return -2 * (y_true - y_pred) / y_true.size;


def cross_entropy(y_pred, y_true):
    raise NotImplementedError

def cross_entropy_d(y_pred, y_true):
    raise NotImplementedError
