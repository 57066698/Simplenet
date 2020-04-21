import numpy as np
from simpleNet.layers.Layer import Layer
import simpleNet.utils.padding_tool as padding_tool
import simpleNet.utils.im2col_tool as im2col_tool
"""
    Conv2d
    
    padding: 跟 torch 一致, 优先补在前面，但不提供手动padding
    -----------------------------
    保存im2col方式的w
    
    x:[N, in_channel, in_h, in_w]
    w:[in_channel x kernel_area, out_channel]
    col:[N, out_h x out_w, in_channel x kernel_area]
    col_out:[N, out_h x out_w, out_channel
    (kernel_area = kernel_h x kernel_w)
    
    dw = d(col_out).T * d(col)
    d(col) = d(col_out) * w.T
    
    -----------------------------
    b = [out_channel]
    db = sum_dims{N, out_h, out_w} da 
"""


class Conv2D(Layer):

    def __init__(self, in_channels:int, out_channels:int, kernel_size, stride:int=1, padding:str= 'valid', use_bias:bool=True):
        assert padding in ['same', 'valid']
        assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple)
        assert in_channels > 0 and out_channels > 0
        super().__init__()

        self.name = "Conv2D"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.use_bias = use_bias

        if isinstance(kernel_size, int):
            self._kernel_h = self._kernel_w = kernel_size
        else:
            self._kernel_h, self._kernel_w = kernel_size

        if isinstance(stride, int):
            self._stride_h = self._stride_w = stride
        else:
            self._stride_h, self._stride_w = stride

        w = np.random.rand(1, self.in_channels * self._kernel_h * self._kernel_w, self.out_channels)
        self.weights = {"w": w}
        if self.use_bias:
            b = np.random.rand(out_channels)
            self.weights["b"] = b

    def __call__(self, *args, **kwargs):
        """
            x -> padding -> col -> col_out -> y -> + b
        """
        x = args[0]
        # 确定out_h, out_w
        N, in_channel, in_h, in_w = x.shape
        assert in_channel == self.in_channels

        if self.padding == "valid":
            out_h = int((in_h - self._kernel_h + 1) / self._stride_h)
            out_w = int((in_h - self._kernel_w + 1) / self._stride_w)
            self.cached_padding = 0
        else:  # padding = same
            out_h = int(np.ceil(in_h/self._stride_h))
            out_w = int(np.ceil(in_w/self._stride_w))

            self.cached_padding = (out_h * self._stride_h + (self._kernel_h - 1) - in_h \
                                   , out_w * self._stride_w + (self._kernel_w - 1) - in_w)

        x = padding_tool.padding_2d(x, self.cached_padding)
        self.cached_x_pad_size = x.shape[2:4]

        col = im2col_tool.im2col(x, self._kernel_h, self._kernel_w, self._stride_h, self._stride_w)
        self.cached_col = col

        col_out = np.matmul(col, self.weights["w"])
        col_out_T = np.transpose(col_out, [0, 2, 1])
        Y = np.reshape(col_out_T, (N, self.out_channels, out_h, out_w))

        if self.use_bias:
            Y[:, ...] = Y[:, ...] + np.reshape(self.weights["b"], (1, self.out_channels, 1, 1))

        return Y

    def backwards(self, dz):
        """
            dz -> db
               -> d(col_out) -> dw
                             -> d(col_in) -> dx_pad -> dx
        """

        self.cached_grad = {}

        N, out_channels, out_h, out_w = dz.shape

        if self.use_bias:
            db = np.sum(dz, axis=(0, 2, 3))
            self.cached_grad["b"] = db

        dcol_out_T = np.reshape(dz, (N, out_channels, -1))
        dcol_out = np.transpose(dcol_out_T, (0, 2, 1))
        dw__ = np.matmul(dcol_out_T, self.cached_col)
        dw_ = np.sum(dw__, axis=0)
        dw = np.transpose(dw_, (1, 0))
        self.cached_grad["w"] = dw
        dcol_in = np.matmul(dcol_out, self.weights['w'].transpose(0, 2, 1))
        N, OHW, _ = dcol_in.shape
        dcol_in = np.reshape(dcol_in, (N, OHW, self.in_channels, self._kernel_h, self._kernel_w))

        (x_pad_h, x_pad_w) = self.cached_x_pad_size
        dx_pad = np.zeros((N, self.in_channels, x_pad_h, x_pad_w), dtype=np.float32)

        # 以 OHW 轴顺序, 从 dcol_in 解出 [N, IC, KH, KW] 的块 加到对应 [i, j] 起头的 dx_pad
        for i in range(out_h):
            for j in range(out_w):
                ohw = i * out_w + j
                col_area = dcol_in[:, ohw, :, :, :]
                top = i * self._stride_h
                bottom = top + self._kernel_h
                left = j * self._stride_w
                right = left + self._kernel_w
                dx_pad[:, :, top: bottom, left: right] += col_area

        dx = padding_tool.depadding_2d(dx_pad, self.cached_x_pad_size)
        return dx

    def __str__(self):
        return "%s: " % self.name + \
               "in_channels: %d, " % self.in_channels + \
               "out_channels: %d, " % self.out_channels + \
               "kernel_size: %s, " % str((self._kernel_h, self._kernel_w)) + \
               "stride: %s, " % str((self._stride_h, self._stride_w)) + \
               "padding: %str, " % self.padding + \
               "use_Bias: %s" % str(self.use_bias)
