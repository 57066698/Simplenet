import numpy as np


def cal_padding_value(x_shape, kernel_size:tuple, stride_size:tuple, padding:str= "valid"):
    """
    从输入尺寸计算padding
    :param x_shape:
    :param kernel_size:
    :param stride_size:
    :param padding:
    :return:
    """

    N, in_channels, in_h, in_w = x_shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride_size

    assert padding in ["valid", "same"]
    if padding == "valid":
        out_h = int((in_h - kernel_h + 1) / stride_h)
        out_w = int((in_h - kernel_w + 1) / stride_w)
        padding_value = 0
    else:  # padding = same
        out_h = int(np.ceil(in_h / stride_h))
        out_w = int(np.ceil(in_w / stride_w))
        in_pad_h = (out_h - 1) * stride_h + kernel_h
        in_pad_w = (out_w - 1) * stride_w + kernel_w
        padding_value = (in_pad_h - in_h, in_pad_w - in_w)

    return padding_value, (out_h, out_w)


def cal_padding_back(out_shape, kernel_size, stride_size):
    """
    从输出尺寸计算padding, 只支持padding same
    :param out_shape:
    :param kernel_size:
    :param stride_size:
    :return:
    """
    N, out_channels, out_h, out_w = out_shape
    stride_h, stride_w = stride_size
    kernel_h, kernel_w = kernel_size
    in_h, in_w = out_h * stride_h, out_w * stride_w
    padding_inner = (stride_h - 1, stride_w - 1)
    padding_out = (kernel_h + stride_h - 2, kernel_w + stride_w - 2)
    return (padding_out, padding_inner), (in_h, in_w)


def padding_2d(x, padding_out, padding_inner = None):
    """
    优先pad 左和上
    :param x:
    :param value: int, 或者 (h/2, h/2, w/2, w/2), 或者 (上, 下, 左, 右)
    :return: pad_x
    """
    # out 参数
    if isinstance(padding_out, int):
        top, bottom, left, right = (padding_out, padding_out, padding_out, padding_out)
    elif len(padding_out) == 2:
        (h, w) = padding_out
        top = int(np.ceil(h/2))
        bottom = h - top
        left = int(np.ceil(w/2))
        right = w - left
    elif len(padding_out) == 4:
        top, bottom, left, right = padding_out
    # inner 参数
    if padding_inner:
        if isinstance(padding_inner, int):
            inner_h, inner_w = padding_inner, padding_inner
        else:
            inner_h, inner_w = padding_inner

    N, channels, in_h, in_w = x.shape

    if padding_inner:
        x_pad_inner = np.zeros((N, channels, (in_h - 1) * inner_h + in_h, (in_w - 1) * inner_w + in_w))
        x_pad_inner[:, :, ::inner_h + 1, ::inner_w + 1] = x[:, :, :, :]
        x = x_pad_inner

    x = np.pad(x, [(0, 0), (0, 0), (top, bottom), (left, right)], mode='constant')

    return x


def depadding_2d(x_pad, padding_out, padding_inner = None):
    """
        移除padding， padding2d的反向操作
        :param x_pad:
        :param value: 之前使用的value
        :return: x
    """

    if isinstance(padding_out, int):
        top, bottom, left, right = (padding_out, padding_out, padding_out, padding_out)
    elif len(padding_out) == 2:
        (h, w) = padding_out
        top = int(np.ceil(h / 2))
        bottom = h - top
        left = int(np.ceil(w / 2))
        right = w - left
    elif len(padding_out) == 4:
        top, bottom, left, right = padding_out

    if padding_inner:
        if padding_inner:
            if isinstance(padding_inner, int):
                inner_h, inner_w = padding_inner, padding_inner
            else:
                inner_h, inner_w = padding_inner

    N, C, H, W = x_pad.shape

    x_pad_inner = x_pad[:, :, top:H - bottom, left:W - right]

    if padding_inner:
        x = x_pad_inner[:, :, ::inner_h+1, ::inner_w+1]
        return x
    else:
        return x_pad_inner
