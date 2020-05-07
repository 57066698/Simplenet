from simpleNet.layers.Layer import Layer
import numpy as np


class Dropout(Layer):
    def __init__(self, p: float == 0.5):
        super().__init__()
        self.name = "Dropout"
        self.drop_rate = p

    def __call__(self, *args, **kwargs):

        if self.mode == "test":
            return args[0]

        x = args[0]
        d = np.random.rand(*x.shape)
        d = d > self.drop_rate
        self.cached_d = d
        x = x * d
        x = x / (1 - self.drop_rate)
        return x

    def backwards(self, da):
        return da * self.cached_d

    def __str__(self):
        return "%s: " % self.name + "drop_rate: %d" % self.drop_rate
