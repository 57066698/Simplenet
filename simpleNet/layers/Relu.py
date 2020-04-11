from simpleNet.layers.Layer import Layer
import numpy as np


class Relu(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        x = args[0]
        self.cached_x = x
        x = np.maximum(0, x)
        return x

    def backwards(self, da):
        da_prev = da.copy()
        da_prev[self.cached_x < 0] = 0
        return da_prev
