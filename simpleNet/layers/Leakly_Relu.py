from simpleNet.layers.Layer import Layer
import numpy as np


class Leakly_Relu(Layer):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.name = "Leakly_Relu"
        self.negative_slope = negative_slope

    def __call__(self, x):
        inds = x > 0
        self.cached_inds = inds
        x[inds == 0] *= self.negative_slope
        return x

    def backwards(self, da):
        da_prev = da.copy()
        da_prev[self.cached_inds == 0] *= self.negative_slope
        return da_prev

    def __str__(self):
        return "%s: negative_slope: %d" % (self.name, self.negative_slope)
