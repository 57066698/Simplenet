import numpy as np


def im2col(input_data, kernel_h, kernel_w, stride_h, stride_w):
    """
        不支持padding
    """
    # 检查和参数
    N, in_channel, H, W = input_data.shape
    out_h, out_w = int(H - (kernel_h-1) / stride_h), int(W - (kernel_w-1) / stride_w)
    assert out_h % 1 == 0 and out_w % 1 == 0
    out_h = int(out_h)
    out_w = int(out_w)

    # 填充 col
    col = np.zeros((N, out_h * out_w, in_channel * kernel_h * kernel_w))

    for block_i in range(out_h):
        for block_j in range(out_w):
            # block_area -> block_data : [N, C, kernel_h, kernel_w] -> [N, C*kernel_h*kernerl_w]
            block_area = input_data[:, :, block_i*stride_h:block_i*stride_h+kernel_h, block_j*stride_w:block_j*stride_w+kernel_w]
            block_data = np.reshape(block_area, (N, in_channel*kernel_h*kernel_w))
            # col_line = block_i * out_w + block_j
            col[:, block_i * out_w + block_j] = block_data

    return col


def col_out2im(col_out, N, out_h, out_w, out_channel):
    col_out_T = np.transpose(col_out, (0, 2, 1))
    im = np.reshape(col_out_T, (N, out_channel, out_h, out_w))
    return im


def w2col(w):
    OC, IC, H, W = w.shape
    return np.reshape(w, (1, IC * H * W, OC))


def col2im(col, out_h, out_w, stride_h, stride_w, kernel_h, kernel_w, in_channel):
    """
    im2col 的逆操作
    :param col:
    :param stride_h:
    :param stride_w:
    :param kernel_h:
    :param kernel_w:
    :param in_channel:
    :return:
    """
    N = col.shape[0]
    im_h = (out_h - 1) * stride_h + kernel_h
    im_w = (out_w - 1) * stride_w + kernel_w
    im = np.zeros((N, in_channel, im_h, im_w))
    for i in range(out_h * out_w):
        row = col[:, i, :]
        h_start = (i // out_w) * stride_h
        w_start = (i % out_w) * stride_w
        im[:, :, h_start: h_start + kernel_h, w_start: w_start + kernel_w] += np.reshape(row, (N, in_channel, kernel_h, kernel_w))
    return im

#
# def col2im(col_data, wcol_h, wcol_w, )
