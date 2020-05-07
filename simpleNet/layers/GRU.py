from simpleNet.layers.Layer import Layer
import simpleNet.utils.actives as actives
import numpy as np

"""
    GRU

    复位门, 更新门, 回忆门
    r[t] = X[t] x wx_r + H[t-1] x wh_r + b_r        -->  tanh
    u[t] = X[t] x wx_u + H[t-1] x wh_u + b_u        -->  tanh
    m[t] = X[t] x wx_m + r[t]*H[t-1] x wh_m + b_m   -->  tanh

    输出层

    H[t] = u[t]*r[t] + (1-u[t])*m[t]

"""


class GRU(Layer):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.name = "GRU"

        self.input_size = input_size
        self.hidden_size = hidden_size

        wx = np.random.rand(input_size, 3 * hidden_size)  # r, u, m 顺序 3 个门
        wh = np.random.rand(hidden_size, 3 * hidden_size)
        b = np.random.rand(3 * hidden_size)

        H0 = np.zeros(hidden_size)

        self.weights = {"wx": wx, "wh": wh, "b": b, "H0": H0}

    def __call__(self, x):
        """
        输出等长度的y
        :param x:
        :return:
        """
        N, length, x_dim = x.shape
        input_size, hidden_size = self.input_size, self.hidden_size
        wx, wh, b = self.weights["wx"], self.weights["wh"], self.weights["b"]
        H0 = self.weights["H0"]

        assert input_size == x_dim

        caches = [None] * length
        H = np.zeros((N, length, hidden_size))
        H[:, -1, :] = H0  # 当i = 0 时 i-1刚好从队尾取到

        for i in range(length):
            H[:, i, :], caches[i] = self.step_forward(x[:, i, :], H[:, i - 1, :], wx, wh, b)

        self.caches = caches
        return H

    def step_forward(self, x, h_prev, wx, wh, b):
        """

        :param x: [N, in]
        :param h_prev: [N, out]
        :param wx: [in, 3 * out]
        :param wh: [out, 3 * out]
        :param b: [3 * out]
        :return:
        """

        N, in_size = x.shape
        in_size, H = self.input_size, self.hidden_size

        # 1  r and u
        mul = np.matmul(x, wx[:, :2*H]) + np.matmul(h_prev, wh[:, :2*H]) + b[:2*H].T

        # 2 reset [N, out]
        a_r = actives.sigmoid(mul[:, 0:H])

        # 3 update
        a_u = actives.sigmoid(mul[:, H:])

        # 4 memery
        m = x.dot(wx[:, 2*H:]) + (a_r * h_prev).dot(wh[:, 2*H:]) + b[2*H:].T
        a_m = actives.tanh(m)

        # 5 output h
        h = a_u * h_prev + (1 - a_u) * a_m

        cache = (x, h_prev.copy(), mul, m, a_r, a_u, a_m, wx, wh, b)
        return h, cache

    def step_backward(self, dh, cache):
        """
        ds 有2个来源
        :param ds: [N, out]
        :param dh: [N, out]
        :param cache:
        :return:
        """
        N, H = dh.shape
        x, h_prev, mul, m, a_r, a_u, a_m, wx, wh, b = cache

        dmul = np.zeros(mul.shape)  # r, u
        dx = np.zeros_like(x)
        dh_prev = np.zeros_like(h_prev)
        dwx = np.zeros_like(wx)  # r, u, m
        dwh = np.zeros_like(wh)  # r, u, m
        db = np.zeros_like(b)    # r, u, m

        # 5
        dh = dh.copy()
        da_u_1 = h_prev * dh
        dh_prev += a_u * dh
        da_m = (1-a_u) * dh
        da_u_2 = - a_m * dh
        da_u = da_u_1 + da_u_2

        # 4
        dm = actives.d_tanh(m) * da_m  # [N, hidden]
        dx += dm.dot(wx[:, 2*H:].T)
        dwx[:, 2*H:] = x.T.dot(dm)  # [in, hidden]  第三部分
        dwh[:, 2*H:] = (a_r * h_prev).T.dot(dm)  # [in, hidden] 第三部分
        db[2*H:] = np.sum(dm, axis=0)
        darh = dm.dot(wh[:, 2*H:].T)
        da_r = h_prev * darh
        dh_prev += a_r * darh  # 后面加到一起

        # 3~2
        dmul[:, :H] = actives.d_sigmoid(a_r) * da_r
        dmul[:, H:] = actives.d_sigmoid(a_u) * da_u

        # 1
        dx += dmul.dot(wx[:, :2*H].T)
        dwx[:, :2*H] = x.T.dot(dmul)
        dh_prev += dmul.dot(wh[:, :2*H].T)
        dwh[:, :2*H] = h_prev.T.dot(dmul)
        db[:2*H] = np.sum(dmul, axis=0)

        return dx, dh_prev, dwx, dwh, db

    def backwards(self, da):
        N, length, out_size = da.shape
        caches, input_size, hidden_size = self.caches, self.input_size, self.hidden_size

        assert out_size == hidden_size
        assert N, length == self.length

        dx = np.zeros((N, length, input_size))
        dH = da
        dwx = np.zeros((input_size, 3 * hidden_size))
        dwh = np.zeros((hidden_size, 3 * hidden_size))
        db = np.zeros(3 * hidden_size)

        dh_prev_ = np.zeros((N, hidden_size))

        for i in reversed(range(length)):
            dH[:, i, :] += dh_prev_
            dx_, dh_prev_, dwx_, dwh_, db_ = self.step_backward(dH[:, i, :], caches[i])
            dx[:, i, :] += dx_
            dwx += dwx_
            dwh += dwh_
            db += db_

        # 此时i=0
        dH0 = np.sum(dh_prev_, axis=0)

        self.cached_grad = {"b": db, "wx": dwx, "wh": dwh, "H0": dH0}
        return dx

    def __repr__(self):
        return "%s : input_size: %s, hidden_size: %s" % (self.name, str(self.input_size), str(self.hidden_size))
