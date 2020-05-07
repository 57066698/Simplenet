import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(y):
    return y * (1-y)


def tanh(x):
    return np.tanh(x)


def d_tanh(s):
    return 1 - np.square(np.tanh(s))

