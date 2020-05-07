from simpleNet.layers.Layer import Layer
import numpy as np


class Tanh(Layer):
    def __init__(self):
        super().__init__()
        self.name = "Tanh"

    def __call__(self, *args, **kwargs):
        x = args[0]
        y = (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))
        self.cached_y = y
        return y

    def backwards(self, da):
        dx = 1 - self.cached_y ** 2
        return dx

    def __str__(self):
        return self.name
