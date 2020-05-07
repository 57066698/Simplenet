from simpleNet.layers.Layer import Layer
import numpy as np


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.name = "Sigmoid"

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def backwards(self, da):
        return da * (1-da)

    def __str__(self):
        return self.name
