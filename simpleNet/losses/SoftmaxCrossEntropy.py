import numpy as np

"""
    经测试等同torch的CategoryCrossEntropy, 除 N 的同时还要除 C
"""
class SoftmaxCrossEntropy:

    def __init__(self, axis = -1):
        self.last_y_true = None
        self.last_y_pred = None
        self.axis = axis

    def __call__(self, y_pred, y_true):
        # y_pred [batch, class, ...]
        # y_true [batch, class, ...]

        exp_x = np.exp(y_pred - np.max(y_pred, axis=self.axis, keepdims=True))
        softmax = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)

        epsilon = 1e-12
        predictions = np.clip(softmax, epsilon, 1. - epsilon)
        ce = -np.sum(y_true * np.log(predictions)) / (y_pred.shape[0] * y_pred.shape[1])

        self.grad = (softmax - y_true) / (y_pred.shape[0] * y_pred.shape[1])
        return ce


        # left = - np.sum(y_pred * y_true)
        #
        # right_batch = np.sum(np.exp(y_pred), axis=1)
        # right_log_batch = np.log(right_batch)
        # right = np.sum(right_log_batch)
        #
        # shape = y_pred.shape
        #
        # l = (left + right) / (shape[0] * shape[2])
        # return l

    def backwards(self):
        return self.grad
