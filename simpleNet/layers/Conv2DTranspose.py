import numpy as np
from simpleNet.layers.Layer import Layer
import simpleNet.utils.padding_tool as padding_tool
import simpleNet.utils.im2col_tool as im2col_tool

"""
    Conv2dTranspose
    按照conv2d的参数设置，将conv2d的输出还原到输入

    padding: 只支持same , 和torch 一致优先补在前面
    out = in * stride
    所以输入像素之间的 padding_inner = stride - 1
    边缘的 padding_border = kernel - 1
    
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


class Conv2DTranspose(Layer):

    def __init__(self, in_channel: int, out_channel: int, kernel_size, stride=1, padding: str = 'same',
                 bias: bool = True):
        assert padding in ['same']
        assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple)
        assert in_channel > 0 and out_channel > 0
        super().__init__()

        self.name = "Conv2DTranspose"
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.padding = padding
        self.use_bias = bias

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride_size = (stride, stride)
        else:
            self.stride_size = stride

        ni = 0.5 * (in_channel + out_channel) * self.kernel_size[0] * self.kernel_size[1]
        w = np.random.normal(0, 2/ni,
                             (1, self.in_channel * self.kernel_size[0] * self.kernel_size[1], self.out_channel))

        self.weights = {"w": w}
        if self.use_bias:
            b = np.zeros(out_channel)
            self.weights["b"] = b

    def __call__(self, x):
        """
            x -> padding -> col -> col_out -> y -> + b
        """
        self.cached_x = x

        padding = self.padding
        kernel_h, kernel_w = self.kernel_size
        in_channel, out_channel = self.in_channel, self.out_channel
        stride_h, stride_w = self.stride_size
        w_col = self.weights['w']
        b = self.weights['b'] if self.use_bias else None
        use_bias = self.use_bias

        # 确定out_h, out_w
        N, IC, in_h, in_w = x.shape
        assert IC == in_channel

        (padding_out, padding_inner), (out_h, out_w) = padding_tool.cal_padding_back(x.shape, (kernel_h, kernel_w),
                                                                       (stride_h, stride_w))

        x_shape = x.shape
        x = padding_tool.padding_2d(x, padding_out, padding_inner)

        col = im2col_tool.im2col(x, kernel_h, kernel_w, 1, 1)

        col_out = np.matmul(col, w_col)

        z = im2col_tool.col_out2im(col_out, N, out_h, out_w, out_channel)

        if use_bias:
            z[:, ...] = z[:, ...] + np.reshape(b, (1, out_channel, 1, 1))

        self.cached_x_shape = x_shape
        self.cached_padding_value = (padding_out, padding_inner)
        self.cached_col = col

        return z

    def backwards(self, dz):
        """
            dz -> db
               -> d(col_out) -> dw
                             -> d(col_in) -> dx_pad -> dx
        """

        use_bias = self.use_bias
        cached_col = self.cached_col
        w = self.weights['w']
        out_channel, in_channel = self.out_channel, self.in_channel
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride_size
        padding_value = self.cached_padding_value
        N, in_channel, in_h, in_w = self.cached_x_shape

        cached_grad = {}

        N, OC, out_h, out_w = dz.shape
        assert OC == out_channel

        # db
        if use_bias:
            db = np.sum(dz, axis=(0, 2, 3))
            cached_grad["b"] = db

        # dw
        dcol_out_T = np.reshape(dz, (N, out_channel, out_h * out_w))
        dcol_out = np.transpose(dcol_out_T, (0, 2, 1))
        dw = np.matmul(cached_col.transpose((0, 2, 1)), dcol_out)
        dw = np.sum(dw, axis=0, keepdims=True)
        cached_grad["w"] = dw

        # dx
        dcol_in = np.matmul(dcol_out, w.transpose(0, 2, 1))
        dcol_in = np.reshape(dcol_in, (N, out_h * out_w, in_channel * kernel_h * kernel_w))
        dx_pad = im2col_tool.col2im(dcol_in, out_h, out_w, 1, 1, kernel_h, kernel_w, in_channel)
        dx = padding_tool.depadding_2d(dx_pad, *padding_value)

        self.cached_grad = cached_grad

        return dx

    def __str__(self):
        return "%s: " % self.name + \
               "in_channels: %d, " % self.in_channel + \
               "out_channels: %d, " % self.out_channel + \
               "kernel_size: %s, " % str(self.kernel_size) + \
               "stride: %s, " % str(self.stride_size) + \
               "padding: %str, " % self.padding + \
               "bias: %s" % str(self.use_bias)
