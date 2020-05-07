from simpleNet.layers.Layer import Layer
import numpy as np

"""
    rnn
    S[t] = S[t-1] x Ws + X[t] x Wx + b
    Y[t] = S[t]
"""

class Rnn(Layer):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.name = "RNN"

        self.input_size = input_size
        self.hidden_size = hidden_size

        wx = np.random.rand(input_size, hidden_size)
        ws = np.random.rand(hidden_size, hidden_size)
        b = np.random.rand(hidden_size)
        S0 = np.zeros((hidden_size))

        self.weights = {"wx": wx, "ws": ws, "b": b, "S0":S0}

    def __call__(self, x):
        """
        输出等长度的y
        :param x:
        :return:
        """
        N, length, x_dim = x.shape
        input_size, hidden_size = self.input_size, self.hidden_size
        wx, ws, b, S0 = self.weights["wx"], self.weights["ws"], self.weights["b"], self.weights["S0"]

        assert input_size == x_dim

        caches = [None] * length
        S = np.zeros((N, length, hidden_size), dtype=x.dtype)
        S[:, -1, :] = S0 # 当i = 0 时 i-1刚好从队尾取到
        for i in range(length):
            S[:, i, :], caches[i] = self.step_forward(x[:, i, :], S[:, i-1, :], wx, ws, b)

        self.caches = caches
        return S

    def step_forward(self, x, s_prev, wx, ws, b):
        xwx = np.dot(x, wx)
        sws = np.dot(s_prev, ws)
        z = b.T + sws + xwx
        s = np.tanh(z)

        cache = (x, s_prev.copy(), wx, ws, z)
        return s, cache

    def step_backward(self, ds, cache):
        x, s_prev, wx, ws, z = cache # [N, in], [N, hidden], [in, hidden], [hidden, hidden], [N, hidden]
        dx, ds_prev, dwx, dws = None, None, None, None # [N, in], [N, hidden], [in, hidden], [hidden, hidden]

        dz = (1 - np.square(np.tanh(z))) * ds

        dxwx = dz  # [N, hidden]
        dsws = dz

        db = np.sum(dz, axis=0)

        dx = np.matmul(dxwx, wx.transpose(1, 0))
        dwx = np.matmul(x.transpose(1, 0), dxwx)

        ds_prev = np.matmul(dsws, ws.transpose(1, 0))
        dws = s_prev.T.dot(dsws)

        return dx, ds_prev, dwx, dws, db

    def backwards(self, da):
        N, length, out_size = da.shape
        caches, input_size, hidden_size = self.caches, self.input_size, self.hidden_size
        wx, ws, b, S0 = self.weights["wx"], self.weights["ws"], self.weights["b"], self.weights["S0"]

        assert out_size == hidden_size
        assert N, length == self.length

        dx = np.zeros((N, length, input_size))
        dwx = np.zeros((input_size, hidden_size))
        dws = np.zeros((hidden_size, hidden_size))
        ds_prev_ = np.zeros((N, hidden_size))
        db = np.zeros(hidden_size)

        dS = da # 同时接受下状态和y 的反向传播

        for i in reversed(range(length)):
            dS[:, i, :] += ds_prev_
            dx_, ds_prev_, dwx_, dws_, db_ = self.step_backward(dS[:, i, :], caches[i])
            dx[:, i, :] += dx_
            dwx += dwx_
            dws += dws_
            db += db_

        # 此时i=0
        dS0 = np.sum(ds_prev_, axis=0)

        self.cached_grad = {"b": db, "wx": dwx, "ws": dws, "S0": dS0}
        return dx

    def __repr__(self):
        return "%s : input_size: %s, hidden_size: %s" % (self.name, str(self.input_size), str(self.hidden_size))
