import numpy as np

"""
    categorical cross entropy
    反向传播是和 softmax 联立的，不能单独使用
"""


class CategoricalCrossEntropy:

    def __init__(self):
        self.last_y_true = None
        self.last_y_pred = None

    def __call__(self, *args, **kwargs):
        self.last_y_pred = args[0]
        self.last_y_true = args[1]
        epsilon = 1e-12
        predictions = np.clip(self.last_y_pred, epsilon, 1. - epsilon)
        ce = -np.sum(self.last_y_true * np.log(predictions)) / predictions.shape[0]
        return ce

    def backwards(self):
        return self.last_y_pred - self.last_y_true
