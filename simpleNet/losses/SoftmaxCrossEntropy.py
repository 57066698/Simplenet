import numpy as np

"""
    多分类loss
    经测试等同torch的CategoryCrossEntropy
    除 N 的同时还要除 C
"""
class SoftmaxCrossEntropy:

    def __init__(self, axis = -1):
        self.last_y_true = None
        self.last_y_pred = None
        self.axis = axis

    def __call__(self, y_pred, y_true):

        exp_x = np.exp(y_pred - np.max(y_pred, axis=self.axis, keepdims=True))
        softmax = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)

        epsilon = 1e-12
        predictions = np.clip(softmax, epsilon, 1. - epsilon)
        ce = -np.sum(y_true * np.log(predictions)) / (y_pred.shape[0] * y_pred.shape[1])

        self.grad = (softmax - y_true) / (y_pred.shape[0] * y_pred.shape[1])
        return ce

    def backwards(self):
        return self.grad
