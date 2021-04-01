import numpy as np

"""
    方差loss
"""
class MeanSquaredError:

    def __init__(self):
        self.last_y_true = None
        self.last_y_pred = None

    def __call__(self, *args, **kwargs):
        self.last_y_pred = args[0]
        self.last_y_true = args[1]
        return np.sum(np.square(self.last_y_pred - self.last_y_true)) / (len(self.last_y_true))

    def backwards(self):
        return (self.last_y_pred - self.last_y_true) / len(self.last_y_true)
