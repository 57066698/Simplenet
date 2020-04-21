import numpy as np


a = np.random.rand(3, 5)
y = np.random.rand(3, 5)


def crossEntropy(y_pred, y):
    logprop = - y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)
    return np.sum(logprop) / y_pred.shape[0]


def crossEntropy2(y_pred, y):
    epsilon = 1e-12
    predictions = np.clip(y_pred, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(y * np.log(predictions + 1e-9)) / N
    return ce


print(crossEntropy(a, y))
print(crossEntropy2(a, y))