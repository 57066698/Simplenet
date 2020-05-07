from simpleNet.layers.Layer import Layer
import numpy as np


class Reshape(Layer):
    def __init__(self, shape):
        super().__init__()
        self.name = "Reshape"
        self.dst_shape = shape

    def __call__(self, *args, **kwargs):
        x = args[0]
        self.src_shape = x.shape
        return np.reshape(x, self.dst_shape)

    def backwards(self, da):
        return np.reshape(da, self.src_shape)

    def __str__(self):
        return self.name
