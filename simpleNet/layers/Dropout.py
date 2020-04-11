from simpleNet.layers.Layer import Layer
import numpy as np


class Dropout(Layer):
    def __init__(self, p:float == 0.5):
        super().__init__()
        self.p = p

    def __call__(self, *args, **kwargs):

        if self.status == "run":
            return args[0]

        x = args[0]
        d = np.random.rand(*x.shape)
        d = d > self.p
        self.cached_d = d
        x = x * d
        x = x / (1-self.p)
        return x

    def backwards(self, da):
        return da * self.cached_d
