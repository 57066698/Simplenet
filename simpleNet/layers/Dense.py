import numpy as np
from simpleNet.layers.Layer import Layer


class Dense(Layer):
    """
        最后1个维度用来做fully_connect
    """

    def __init__(self, input_num: int, hidden_num: int, bias: bool = True):
        super().__init__()
        self.name = "Dense"
        self.hidden_num = hidden_num
        self.bias = bias

        w = np.random.normal(0, 2/hidden_num, (input_num, self.hidden_num))
        self.weights = {"w":w}
        if self.bias:
            b = np.zeros(self.hidden_num)
            self.weights["b"] = b

    def __call__(self, *args, **kwargs):
        # forward
        x = args[0]
        self.cached_x = x # 备用
        if self.bias:
            w, b = self.weights["w"], self.weights["b"]
            z = np.matmul(x, w) + b
        else:
            w = self.weights["w"]
            z = np.matmul(x, w)
        return z

    def backwards(self, da):
        x = self.cached_x
        axis = np.arange(len(x.shape))
        axis[-2:] = [axis[-1], axis[-2]]

        x_T = np.transpose(x, tuple(axis.tolist()))
        dw = np.matmul(x_T, da)
        dw = np.sum(dw, axis=tuple(axis[:-2].tolist()))
        self.cached_grad = {"w": dw}
        if self.bias:
            db = np.sum(da.reshape(-1, self.hidden_num), axis=0)
            self.cached_grad["b"] = db
        w_T = self.weights["w"].transpose()
        a_prev = np.matmul(da, w_T)
        return a_prev

    def __str__(self):
        return "%s: " % self.name + "hidden: %d, " % self.hidden_num + "bias: %s" % str(self.bias)