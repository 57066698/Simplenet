from simpleNet.layers.Layer import Layer
import numpy as np


class Transpose(Layer):
    def __init__(self, dst_axis):
        super().__init__()
        self.name = "Transpose"
        num_axis = len(dst_axis)
        trans = np.arange(num_axis) - np.array(dst_axis)
        back = - trans
        back_axis = np.zeros(num_axis)
        back_axis[np.arange(num_axis) + back] = np.arange(num_axis)
        self.dst_axis = dst_axis
        self.back_axis = tuple(back_axis.astype(np.int).tolist())

    def __call__(self, x):
        assert len(x.shape) == len(self.dst_axis)
        y = np.transpose(x, self.dst_axis)
        return y

    def backwards(self, da):
        assert len(da.shape) == len(self.dst_axis)
        dx = np.transpose(da, self.back_axis)
        return dx

    def __str__(self):
        return "(%s): axis: %s" % (self.name, self.dst_axis)
