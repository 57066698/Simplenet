# 测试 Conv2d
import numpy as np
from simpleNet import layers, optims, losses
import torch
from torch import nn

# # 测试正向
#
# N, C, H, W = 1, 1, 5, 5
# OC, stride = 1, 3
# k_h, k_w = 3, 3
#
# x = np.arange(N*C*H*W) + 1
# x = np.reshape(x, (N, C, H, W))
#
# my_conv = layers.Conv2D(OC, C, stride, padding="same", use_bias=False)
# torch_conv = nn.Conv2d(OC, C, stride, padding=1, bias=False)
# input = torch.Tensor(x)
#
# weight_np = np.arange(OC*C*k_h*k_w)
# weight_np = np.reshape(weight_np, (OC, C, k_h, k_w))
# w_col = weight_np.reshape(1, OC, C * k_h * k_w)
# w_col = np.transpose(w_col, (0, 2, 1))
# my_conv.weights["w"] = w_col
#
# weight_b = np.random.rand(C)
# my_conv.weights["b"] = weight_b
#
# with torch.no_grad():
#     a = torch.Tensor(weight_np)
#     p = torch.nn.Parameter(a)
#     b = torch.Tensor(weight_b)
#     p2 = torch.nn.Parameter(b)
#
#     torch_conv.weight = p
#     # torch_conv.bias = p2
#
#     print(torch_conv(input))
#
# print(my_conv(x))


# 测试反向
conv = layers.Conv2D(2, 1, 2, padding="same", use_bias=True)
x = np.ones((2, 2, 2, 2), dtype=np.float32)
x[1, ...] += 1
y = np.random.rand(2, 1, 2, 2)
y[1, :, :, :] += 1
critterion = optims.Adam(conv)
l = losses.MeanSquaredError()

for i in range(10000):
    y_pred = conv(x)
    L = l(y_pred, y)
    da = l.backwards()
    conv.backwards(da)
    critterion.step()

# print(x)
print(y)
print("--------------------")

# print(conv.weights["w"])
print(conv(x))