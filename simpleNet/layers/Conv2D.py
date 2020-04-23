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

    def __init__(self, in_channel:int, out_channel:int, kernel_size, stride=1, padding:str= 'valid', bias:bool=True):
        assert padding in ['same', 'valid']
        assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple)
        assert in_channel > 0 and out_channel > 0
        super().__init__()

        self.name = "Conv2D"
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

        w = np.random.rand(1, self.in_channel * self.kernel_size[0] * self.kernel_size[1], self.out_channel).astype(np.float64)
        self.weights = {"w": w}
        if self.use_bias:
            b = np.random.rand(out_channel)
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

        padding_value, (out_h, out_w) = padding_tool.cal_padding_value(x.shape, (kernel_h, kernel_w), (stride_h, stride_w), padding)

        x_shape = x.shape
        x = padding_tool.padding_2d(x, padding_value)
        col = im2col_tool.im2col(x, kernel_h, kernel_w, stride_h, stride_w)

        col_out = np.matmul(col, w_col)

        z = im2col_tool.col_out2im(col_out, N, out_h, out_w, out_channel)

        if use_bias:
            z[:, ...] = z[:, ...] + np.reshape(b, (1, out_channel, 1, 1))

        self.cached_x_shape = x_shape
        self.cached_padding_value = padding_value
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
            db = np.sum(dz, axis=(0, 2, 3)) / N
            cached_grad["b"] = db

        # dw
        dcol_out_T = np.reshape(dz, (N, out_channel, out_h * out_w))
        dcol_out = np.transpose(dcol_out_T, (0, 2, 1))
        dw = np.matmul(cached_col.transpose((0, 2, 1)), dcol_out)
        dw = np.sum(dw, axis=0, keepdims=True) / N
        cached_grad["w"] = dw

        # dx
        dcol_in = np.matmul(dcol_out, w.transpose(0, 2, 1))
        dcol_in = np.reshape(dcol_in, (N, out_h * out_w, in_channel * kernel_h * kernel_w))
        dx_pad = im2col_tool.col2im(dcol_in, out_h, out_w, stride_h, stride_w, kernel_h, kernel_w, in_channel)
        dx = padding_tool.depadding_2d(dx_pad, padding_value)

        self.cached_grad = cached_grad

        return dx



    def __str__(self):
        return "%s: " % self.name + \
               "in_channels: %d, " % self.in_channel + \
               "out_channels: %d, " % self.out_channel + \
               "kernel_size: %s, " % str((self.kernel_h, self.kernel_w)) + \
               "stride: %s, " % str((self._stride_h, self._stride_w)) + \
               "padding: %str, " % self.padding + \
               "use_Bias: %s" % str(self.use_bias)


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None

    x, w, b, conv_param = cache
    pad_num = conv_param['pad']
    stride = conv_param['stride']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    H_prime = (H + 2 * pad_num - HH) // stride + 1
    W_prime = (W + 2 * pad_num - WW) // stride + 1

    dw = np.zeros(w.shape)
    dx = np.zeros(x.shape)
    db = np.zeros(b.shape)

    # We could calculate the bias by just summing over the right dimensions
    # Bias gradient (Sum on dout dimensions (batch, rows, cols)
    # db = np.sum(dout, axis=(0, 2, 3))

    for i in range(N):
        im = x[i, :, :, :]
        im_pad = np.pad(im, ((0, 0), (pad_num, pad_num), (pad_num, pad_num)), 'constant')
        im_col = im2col(im_pad, HH, WW, stride)  # [OHxOW, ICxKHxKW]
        filter_col = np.reshape(w, (F, -1)).T

        dout_i = dout[i, :, :, :]
        dbias_sum = np.reshape(dout_i, (F, -1))  # [OC, OHxOW]
        dbias_sum = dbias_sum.T  # [OHxoW, OC]

        # bias_sum = mul + b
        db += np.sum(dbias_sum, axis=0)
        dmul = dbias_sum  # [OHxOW, OC]

        # mul = im_col * filter_col
        dfilter_col = (im_col.T).dot(dmul)  # [ICxKHxKW, OC]
        dim_col = dmul.dot(filter_col.T)  # [OHxOW, ICxKHxKW]

        dx_padded = col2im_back(dim_col, H_prime, W_prime, stride, HH, WW, C)
        dx[i, :, :, :] = dx_padded[:, pad_num:H + pad_num, pad_num:W + pad_num]
        dw += np.reshape(dfilter_col.T, (F, C, HH, WW))
    return dx, dw, db


def im2col(x,hh,ww,stride):

    """
    Args:
      x: image matrix to be translated into columns, (C,H,W)
      hh: filter height
      ww: filter width
      stride: stride
    Returns:
      col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
            new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    """

    c,h,w = x.shape
    new_h = (h-hh) // stride + 1
    new_w = (w-ww) // stride + 1
    col = np.zeros([new_h*new_w,c*hh*ww])

    for i in range(new_h):
       for j in range(new_w):
           patch = x[...,i*stride:i*stride+hh,j*stride:j*stride+ww]
           col[i*new_w+j,:] = np.reshape(patch,-1)
    return col


def col2im_back(dim_col, h_prime, w_prime, stride, hh, ww, c):
    """
    Args:
      dim_col: gradients for im_col,(h_prime*w_prime,hh*ww*c)
      h_prime,w_prime: height and width for the feature map
      strid: stride
      hh,ww,c: size of the filters
    Returns:
      dx: Gradients for x, (C,H,W)
    """
    H = (h_prime - 1) * stride + hh
    W = (w_prime - 1) * stride + ww
    dx = np.zeros([c, H, W])
    for i in range(h_prime * w_prime):
        row = dim_col[i, :]
        h_start = int((i / w_prime) * stride)
        w_start = int((i % w_prime) * stride)
        dx[:, h_start:h_start + hh, w_start:w_start + ww] += np.reshape(row, (c, hh, ww))
    return dx


