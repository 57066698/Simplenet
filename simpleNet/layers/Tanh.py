from simpleNet.layers.Layer import Layer
import numpy as np


class Tanh(Layer):
    def __init__(self):
        super().__init__()
        self.name = "Tanh"

    def __call__(self, x):
        a = 2 / (1 + np.exp(-2 * x)) - 1
        self.cached_a = a
        return a

    def backwards(self, da):
        z = (1 - self.cached_a ** 2) * da
        return z

    def __str__(self):
        return self.name
