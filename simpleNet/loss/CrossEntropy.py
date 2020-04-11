import numpy as np


class CrossEntropy:

    def __init__(self):
        self.last_y_true = None
        self.last_y_pred = None

    def __call__(self, *args, **kwargs):
        self.last_y_pred = args[0]
        self.last_y_true = args[1]
        epsilon = 1e-12
        predictions = np.clip(self.last_y_pred, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = -np.sum(self.last_y_true * np.log(predictions + 1e-9)) / N
        return ce

    def backwards(self):
        return self.last_y_pred - self.last_y_pred
