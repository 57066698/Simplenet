import numpy as np

"""
    categorical cross entropy
    反向传播是和 softmax 联立的，不能单独使用
"""


class CategoricalCrossEntropy:

    def __init__(self):
        self.last_y_true = None
        self.last_y_pred = None

    def __call__(self, y_pred, y_true):
        self.last_y_pred = y_pred
        self.last_y_true = y_true
        epsilon = 1e-12
        predictions = np.clip(self.last_y_pred, epsilon, 1. - epsilon)
        #todo: 这个除只能针对3维情况
        ce = -np.sum(self.last_y_true * np.log(predictions)) / (y_pred.shape[0] * y_pred.shape[1])
        return ce

    def backwards(self):
        return (self.last_y_pred - self.last_y_true) / (self.last_y_pred.shape[0] * self.last_y_pred.shape[1])
