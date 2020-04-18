from simpleNet.layers.Layer import Layer
import numpy as np


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.name = "Flatten"

    def __call__(self, *args, **kwargs):
        x = args[0]
        self.cached_shape = tuple(x.shape)
        x = np.reshape(x, (x.shape[0], -1))
        return x

    def backwards(self, da):
        da_prev = np.reshape(da, self.cached_shape)
        return da_prev

    def __str__(self):
        return self.name
