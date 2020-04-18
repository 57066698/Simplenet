import numpy as np
from simpleNet.layers.Layer import Layer


class Conv2D(Layer):

    def __init__(self, in_channels:int, out_channels:int, kerner_size, stride:int=1, padding:str='valid', use_bias:bool=True):
        assert padding in ['same', 'valid']
        assert isinstance(kerner_size, int) or isinstance(kerner_size, tuple)
        assert in_channels > 0 and out_channels > 0
        super().__init__()
        self.name = "Conv2D"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias

        # w: channels * kerner_size * kerner_size

        if isinstance(kerner_size, int):
            self._kerner_h = self._kerner_w = kerner_size
        else:
            self._kerner_h, self._kerner_w = kerner_size

        w = np.random.rand(self.out_channels, self.in_channels, self._kerner_h, self._kerner_w)
        self.weights = [w]
        if self.use_bias:
            b = np.random.rand(out_channels)
            self.weights.append(b)

    def __call__(self, *args, **kwargs):
        """
            从输出的每个点 O(Oi, Oj, Oc) 对应到源区域 I (Ii, Ij 尺寸为 kerner_h, kerner_w) 的每个通道 Ic
            out = sum Ii,Ij,Ic <- O * w[Oc, Ic, Oi, Oj]
        """
        x = args[0]
        m = x.shape[0]
        # 确定out_h, out_w
        in_h, in_w = x.shape[2:4]

        if self.padding == "valid":
            out_h = int((in_h - self._kerner_h + 1) / self.stride)
            out_w = int((in_h - self._kerner_w + 1) / self.stride)
            self.cached_padding = (0, 0, 0, 0)
        else:
            out_h = int(np.ceil(in_h/self.stride))
            out_w = int(np.ceil(in_w/self.stride))

            padding_h, padding_w = out_h * self.stride + (self._kerner_h - 1) - in_h, out_w * self.stride + (self._kerner_w - 1) - in_w
            padding_top = int(padding_h / 2)
            padding_bottom = padding_h - padding_top
            padding_left = int(padding_w / 2)
            padding_right = padding_w - padding_left

            self.cached_padding = (padding_top, padding_bottom, padding_left, padding_right)

            if padding_top > 0:
                top = np.zeros((m, self.in_channels, padding_top, in_w), dtype=np.float32)
                x = np.concatenate((top, x), axis=2)
            if padding_bottom > 0:
                bottom = np.zeros((m, self.in_channels, padding_bottom, in_w), dtype=np.float32)
                x = np.concatenate((x, bottom), axis=2)
            if padding_left > 0:
                left = np.zeros((m, self.in_channels, in_h + padding_h, padding_left),dtype=np.float32)
                x = np.concatenate((left, x), axis=3)
            if padding_right > 0:
                right = np.zeros((m, self.in_channels, in_h + padding_h, padding_right), dtype=np.float32)
                x = np.concatenate((x, right), axis=3)

        self.cached_x = x # padding也cache进来
        self.cached_in_size = (in_h, in_w)

        m = x.shape[0]
        k_h = self._kerner_h
        k_w = self._kerner_w
        s = self.stride
        W = self.weights[0]
        Y = np.zeros((m, self.out_channels, out_h, out_w))
        # 映射
        for i in range(m):
            for Oc in range(self.out_channels):
                for Oi in range(out_h):
                    for Oj in range(out_w):
                        # 对Ic, Ii, Ij 点乘再求和
                        in_mat = x[i, :, Oi*s: Oi*s + k_h, Oj*s: Oj*s + k_w]
                        w_mat = W[Oc, :, :, :]
                        out_mat = in_mat * w_mat
                        Y[i, Oc, Oi, Oj] = np.sum(out_mat)

            if self.use_bias:
                Y[i, ...] = Y[i, ...] + np.tile(self.weights[1].reshape(self.out_channels, 1, 1), (1, out_h, out_w))

        return Y

    def backwards(self, da):
        # da [batch_size, output_channel, output_h, output_w]
        assert len(da.shape) == 4
        assert da.shape[0] == self.cached_x.shape[0]
        assert da.shape[1] == self.out_channels

        self.cached_grad = []

        (in_h, in_w) = self.cached_x.shape[2:4]
        (out_h, out_w) = da.shape[2:4]
        # 对 dw(Ic,i,j|Oc)， 其值为 x[batch_size,Ic,h,w] 的和 乘以 da

        w = self.weights[0]
        s = self.stride
        k_h = self._kerner_h
        k_w = self._kerner_w
        m = self.cached_x.shape[0]
        grad_w = np.zeros(self.weights[0].shape, dtype=np.float32)  # [in_channels, out_channels, k_h, k_w]
        da_prev = np.zeros(self.cached_x.shape, dtype=np.float32)
        # 都是点乘，从da出发，逐点把梯度加到 grad_w 和 da_prev上即可
        for batch_i in range(m):
            for Oc in range(self.out_channels):
                for Oi in range(out_h):
                    for Oj in range(out_w):
                        # 对于dw 用x对应点集乘上相应的da点集
                        # w对应点是 Oc, :, 0:k_h, 0:k_w
                        # x对应点是 batch_i, :, Oi*s:Oi*s + k_h, Oj*s:Oj*s + k_w
                        # da对应点是 batch_i, Oc, Oi, Oj
                        grad_w[Oc, :, :, :] += da[batch_i, Oc, Oi, Oj] \
                                    * self.cached_x[batch_i, :,  Oi*s:Oi*s + k_h, Oj*s:Oj*s + k_w]
                        # 对于dx 用w对应点集乘上相应的da点集合
                        da_prev[batch_i, :, Oi*s:Oi*s+k_h, Oj*s:Oj*s+k_w] += da[batch_i, Oc, Oi, Oj] \
                                    * w[Oc, :, :, :]

        self.cached_grad.append(grad_w)

        if self.use_bias:
            db = np.sum(da, axis=(0,2,3))
            self.cached_grad.append(db)

        padding_top, padding_down, padding_left, padding_right = self.cached_padding
        return da_prev[:, :, padding_top:-padding_down, padding_left:-padding_right]

    def __str__(self):
        return "%s: " % self.name + \
               "in_channels: %d, " % self.in_channels + \
               "out_channels: %d, " % self.out_channels + \
               "kerner_size: %s, " % str((self._kerner_h, self._kerner_w)) + \
               "strid: %d, " % self.stride if self.stride != 2 else "" + \
               "padding: %str, " % self.padding + \
               "use_Bias: %s" % str(self.use_bias)
