from simpleNet.layers.Layer import Layer
import numpy as np


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.name = "Sigmoid"

    def __call__(self, x):
        a = 1 / (1 + np.exp(-x))
        self.cached_a = a
        return a

    def backwards(self, da):
        a = self.cached_a
        return a * (1-a) * da

    def __str__(self):
        return self.name
