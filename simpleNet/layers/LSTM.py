from simpleNet.layers.Layer import Layer
import simpleNet.utils.actives as actives
import numpy as np

"""
    LSTM
    
    输入, 遗忘门, 输入门, 输出门
    
    in[t] = X[t]*W_in + H(t-1)*U_in + b_in  -->  tanh
    f[t] = X[t]*W_f + H(t-1)*U_f + b_f      -->  sigmoid
    g[t] = X[t]*W_g + H(t-1)*U_g + b_g      -->  sigmoid
    q[t] = X[t]*W_q + H(t-1)*U_q + b_q      -->  sigmoid
    
    记忆层, 输出层
    
    S(t) = f[t]*S[t-1] + g[t]*in[t]             回忆+新知
    H[t] = tanh(St)*q[t]                         输出
    
"""


class LSTM(Layer):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.name = "LSTM"

        self.input_size = input_size
        self.hidden_size = hidden_size

        # torch 初始化 U(-(1/hidden)**0.5, sqrt(1/hidden)**0.5)
        low, high = - (6/(hidden_size+input_size)) ** 0.5, (6/(hidden_size+input_size)) ** 0.5
        wx = np.random.uniform(low, high, (input_size, 4 * hidden_size))  # in, forgot, quit, g 顺序 4 个门
        wh = np.random.uniform(low, high, (hidden_size, 4 * hidden_size))
        b = np.zeros(4 * hidden_size)

        S0 = np.zeros(hidden_size)
        H0 = np.zeros(hidden_size)

        self.weights = {"wx": wx, "wh": wh, "b": b, "S0": S0, "H0": H0}

    def __call__(self, x, init: tuple = None):
        """
        输出等长度的y
        :param x:
        :return:
        """
        N, length, x_dim = x.shape
        input_size, hidden_size = self.input_size, self.hidden_size
        wx, wh, b = self.weights["wx"], self.weights["wh"], self.weights["b"]
        if init:
            assert init[0].shape == (N, hidden_size)
            assert init[1].shape == (N, hidden_size)
            H0, S0 = init
        else:
            H0, S0 = self.weights["H0"], self.weights["S0"]

        assert input_size == x_dim

        caches = [None] * length
        S = np.zeros((N, length, hidden_size), dtype=x.dtype)
        S[:, -1, :] = S0 # 当i = 0 时 i-1刚好从队尾取到
        H = np.zeros_like(S)
        H[:, -1, :] = H0

        for i in range(length):
            S[:, i, :], H[:, i, :], caches[i] = self.step_forward(x[:, i, :], S[:, i-1, :], H[:, i-1, :], wx, wh, b)

        self.caches = caches
        self.cached_hShape = H.shape

        return H, (H[:, -1, :], S[:, -1, :])

    def step_forward(self, x, s_prev, h_prev, wx, wh, b):
        """

        :param x: [N, in]
        :param s_prev: [N, out]
        :param h_prev: [N, out]
        :param wx: [in, 4 * out]
        :param wh: [out, 4 * out]
        :param b: [4 * out]
        :return:
        """

        N, in_size = x.shape
        in_size, H = self.input_size, self.hidden_size

        # 1
        mul = np.matmul(x, wx) + np.matmul(h_prev, wh) + b.T

        # 2 input [N, out]
        a_i = actives.sigmoid(mul[:, 0:H])

        # 3 forget
        a_f = actives.sigmoid(mul[:, H:2*H])

        # 4 output
        a_o = actives.sigmoid(mul[:, 2*H:3*H])

        # 5 memory
        a_g = actives.tanh(mul[:, 3*H:4*H])

        # 6 now state
        s = a_f * s_prev + a_g * a_i

        # 7 output
        h = actives.tanh(s) * a_o

        cache = (x, s_prev.copy(), h_prev.copy(), mul, a_i, a_f, a_o, a_g, s, h, wx, wh)
        return s, h, cache

    def step_backward(self, ds, dh, cache):
        """
        ds 有2个来源
        :param ds: [N, out]
        :param dh: [N, out]
        :param cache:
        :return:
        """
        N, H = dh.shape
        x, s_prev, h_prev, mul, a_i, a_f, a_o, a_g, s, h, wx, wh = cache

        dmul = np.zeros(mul.shape)

        # 7
        ds = ds.copy()
        dtanh_s = a_o * dh
        ds += actives.d_tanh(s) * dtanh_s
        da_o = actives.tanh(s) * dh

        # 6
        da_f = ds * s_prev
        ds_prev = ds * a_f
        da_g = ds * a_i
        da_i = ds * a_g

        # 5 ~ 2
        g = mul[:, 3*H:4*H]
        dmul[:, 3*H:4*H] = actives.d_tanh(g) * da_g
        dmul[:, 2*H:3*H] = actives.d_sigmoid(a_o) * da_o
        dmul[:, 1*H:2*H] = actives.d_sigmoid(a_f) * da_f
        dmul[:, 0*H:1*H] = actives.d_sigmoid(a_i) * da_i

        # 1
        db = np.sum(dmul, axis=0)
        dx = dmul.dot(wx.T)
        dwx = x.T.dot(dmul)
        dh_prev = dmul.dot(wh.T)
        dwh = h_prev.T.dot(dmul)

        return dx, ds_prev, dh_prev, dwx, dwh, db

    def backwards(self, da=None, dhds=None):
        """
        da 和 dhds 至少要由一项
        :param da:
        :param dhds: (dh_last, ds_last)
        :return:
        """

        caches, input_size, hidden_size = self.caches, self.input_size, self.hidden_size
        N, length, out_size = self.cached_hShape

        if da is None:
            da = np.zeros((N, length, out_size))
            assert dhds and len(dhds) == 2
        else:
            assert da.shape == self.cached_hShape

        if dhds:
            dh_prev_, ds_prev_ = dhds
            assert dh_prev_.shape == (N, hidden_size)
            assert ds_prev_.shape == (N, hidden_size)
        else:
            ds_prev_ = np.zeros((N, hidden_size))
            dh_prev_ = np.zeros((N, hidden_size))

        dx = np.zeros((N, length, input_size))
        dS = np.zeros((N, length, hidden_size))
        dH = da
        dwx = np.zeros((input_size, 4*hidden_size))
        dwh = np.zeros((hidden_size, 4*hidden_size))
        db = np.zeros(4*hidden_size)

        for i in reversed(range(length)):
            dS[:, i, :] += ds_prev_
            dH[:, i, :] += dh_prev_
            dx_, ds_prev_, dh_prev_, dwx_, dwh_, db_ = self.step_backward(dS[:, i, :], dH[:, i, :], caches[i])
            dx[:, i, :] += dx_
            dwx += dwx_
            dwh += dwh_
            db += db_

        # 此时i=0
        dS0 = np.sum(ds_prev_, axis=0)
        dH0 = np.sum(dh_prev_, axis=0)

        self.cached_grad = {"b": db, "wx": dwx, "wh": dwh, "S0": dS0, "H0": dH0}
        return dx, (dh_prev_, ds_prev_)

    def __str__(self):
        return "%s : input_size: %s, hidden_size: %s" % (self.name, str(self.input_size), str(self.hidden_size))
