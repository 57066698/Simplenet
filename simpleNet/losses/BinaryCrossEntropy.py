import numpy as np
"""
    binary cross entropy
"""


class CategoricalCrossEntropy:

    def __init__(self):
        self.last_y_true = None
        self.last_y_pred = None

    def __call__(self, y_pred, y_true):
        self.last_y_pred = y_pred
        self.last_y_true = y_true
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        logprobs = - np.log(y_pred) * y_true - np.log(1 - y_pred) * (1 - y_true)
        bce = np.sum(logprobs) / y_pred.shape[0]
        return bce

    def backwards(self):
        y_pred, y_true = self.last_y_pred, self.last_y_true
        return (1 - y_true) / (1 - y_pred) - (y_true / y_pred)
