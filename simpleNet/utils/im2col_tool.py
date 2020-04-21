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
