import numpy as np

# 改自 https: // blog.csdn.net / Daycym / article / details / 83826222


def im2col(input_data, kernel_data, stride_h, stride_w):
    """
        不支持padding
    """
    # 检查和参数
    N, C, H, W = input_data.shape
    OC, IC, kernel_h, kernel_w = kernel_data.shape
    assert IC == C
    out_h, out_w = int(H - (kernel_h-1) / stride_h), int(W - (kernel_w-1) / stride_w)
    assert out_h % 1 == 0 and out_w % 1 == 0
    out_h = int(out_h)
    out_w = int(out_w)

    # 填充 col
    col = np.zeros((N, out_h * out_w, C * kernel_h * kernel_w))

    for block_i in range(out_h):
        for block_j in range(out_w):
            # block_area -> block_data : [N, C, kernel_h, kernel_w] -> [N, C*kernel_h*kernerl_w]
            block_area = input_data[:, :, block_i*stride_h:block_i*stride_h+kernel_h, block_j*stride_w:block_j*stride_w+kernel_w]
            block_data = np.reshape(block_area, (N, C*kernel_h*kernel_w))
            # col_line = block_i * out_w + block_j
            col[:, block_i * out_w + block_j] = block_data

    # 变换kernel
    # kernel_data -> col_kernel : [OC, kernel_h, kernel_w] -> [OC, kernel_h * kernel_w] -> [kernel_h * kernel_w, OC]
    col_kernel = np.reshape(kernel_data, (1, OC, IC * kernel_h * kernel_w))
    col_kernel = np.transpose(col_kernel, (0, 2, 1))
    return col, col_kernel


def col2im(col, col_kerner, kerner_shape,input_shape, stride_h, stride_w):
    N, C, H, W = input_shape
    OC, IC, filter_h, filter_w = kerner_shape
    out_h, out_w = int(H - (filter_h - 1) / stride_h), int(W - (filter_w - 1) / stride_w)
    assert out_h % 1 == 0 and out_w % 1 == 0
    out_h = int(out_h)
    out_w = int(out_w)

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H, W))
    for y in range(filter_h):
        y_max = y + stride_h*out_h
        for x in range(filter_w):
            x_max = x + stride_w*out_w
            img[:, :, y:y_max:stride_h, x:x_max:stride_w] += col[:, :, y, x, :, :]

    return img
