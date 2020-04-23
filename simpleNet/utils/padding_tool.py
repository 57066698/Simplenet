import numpy as np


def cal_padding_value(x_shape, kernel_size:tuple, stride_size:tuple, padding:str= "valid"):
    """
    从输入尺寸，核尺寸，和步长计算pad值和输出尺寸
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


def padding_2d(x, value):
    """
    优先pad 左和上
    :param x:
    :param value: int, 或者 (h/2, h/2, w/2, w/2), 或者 (上, 下, 左, 右)
    :return: pad_x
    """

    if isinstance(value, int):
        top, bottom, left, right = (value, value, value, value)
    elif len(value) == 2:
        (h, w) = value
        top = int(np.ceil(h/2))
        bottom = h - top
        left = int(np.ceil(w/2))
        right = w - left
    elif len(value) == 4:
        top, bottom, left, right = value

    x = np.pad(x, [(0, 0), (0, 0), (top, bottom), (left, right)], mode='constant')

    return x


def depadding_2d(pad_x, value):
    """
        移除padding， padding2d的反向操作
        :param pad_x:
        :param value: 之前使用的value
        :return: x
    """

    if isinstance(value, int):
        top, bottom, left, right = (value, value, value, value)
    elif len(value) == 2:
        (h, w) = value
        top = int(np.ceil(h / 2))
        bottom = h - top
        left = int(np.ceil(w / 2))
        right = w - left
    elif len(value) == 4:
        top, bottom, left, right = value

    N, C, H, W = pad_x.shape

    return pad_x[:, :, top:H-bottom, left:W-right]
