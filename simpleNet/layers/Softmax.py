from simpleNet.layers.Layer import Layer
import numpy as np


class Softmax(Layer):

    def __init__(self, axis=-1):
        super().__init__()
        self.name = "Softmax"
        self.axis = axis

    def __call__(self, x):
        exp_x = np.exp(x - np.max(x, axis=self.axis, keepdims=True))
        out = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)
        return out

    def backwards(self, da):
        ## 只支持 cross-entropy
        return da

    def __str__(self):
        return self.name
