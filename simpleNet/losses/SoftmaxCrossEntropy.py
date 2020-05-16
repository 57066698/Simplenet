import numpy as np


class SoftmaxCrossEntropy:

    def __init__(self):
        self.last_y_true = None
        self.last_y_pred = None

    def __call__(self, y_pred, y_true):
        # y_pred [batch, class, len]
        # y_true [batch, class, len]

        self.last_y_pred = y_pred
        self.last_y_true = y_true

        left = - np.sum(y_pred * y_true)

        right_batch = np.sum(np.exp(y_pred), axis=1)
        right_log_batch = np.log(right_batch)
        right = np.sum(right_log_batch)

        shape = y_pred.shape

        l = (left + right) / (shape[0] * shape[2])
        return l

    def backwards(self):
        return (self.last_y_pred - self.last_y_true) / (self.last_y_pred.shape[0] * self.last_y_pred.shape[1])
