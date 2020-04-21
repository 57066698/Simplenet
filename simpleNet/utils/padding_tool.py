import numpy as np


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

    N, C, H, W = x.shape

    if top > 0:
        top_pad = np.zeros((N, C, top, W), dtype=np.float32)
        x = np.concatenate((top_pad, x), axis=2)
    if bottom > 0:
        bottom_pad = np.zeros((N, C, bottom, W), dtype=np.float32)
        x = np.concatenate((x, bottom_pad), axis=2)
    if left > 0:
        left_pad = np.zeros((N, C, H + top + bottom, left), dtype=np.float32)
        x = np.concatenate((left_pad, x), axis=3)
    if right > 0:
        right_pad = np.zeros((N, C, H + top + bottom, right), dtype=np.float32)
        x = np.concatenate((x, right_pad), axis=3)

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
