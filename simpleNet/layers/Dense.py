import numpy as np
from simpleNet.layers.Layer import Layer


class Dense(Layer):

    def __init__(self, input_num: int, hidden_num: int, use_bias: bool = True):
        super().__init__()
        self.hidden_num = hidden_num
        self.use_bias = use_bias

        w = np.random.rand(input_num, self.hidden_num)
        self.weights = [w]
        if self.use_bias:
            b = np.random.rand(self.hidden_num)
            self.weights.append(b)

    def __call__(self, *args, **kwargs):
        # forward
        x = args[0]
        self.cached_x = x # 备用
        if self.use_bias:
            w, b = self.weights[:2]
            z = np.matmul(x, w) + b
        else:
            w = self.weights[0]
            z = np.matmul(x, w)
        return z

    def backwards(self, da):
        x = self.cached_x
        dw = x.transpose().dot(da)
        self.cached_grad = [dw]
        if self.use_bias:
            db = np.sum(da, axis=0)
            self.cached_grad.append(db)
        w_T = self.weights[0].transpose()
        a_prev = np.matmul(da, w_T)
        return a_prev
