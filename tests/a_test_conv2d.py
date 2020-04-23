# 测试 Conv2d
import numpy as np
from simpleNet import layers
import torch
from torch import nn

# 测试正向

N, C, H, W = 2, 2, 5, 5
OC, stride = 1, 3
k_h, k_w = 3, 3

x = np.arange(N*C*H*W) + 1
x = np.reshape(x, (N, C, H, W))
y = np.random.rand(N, OC, H, W)

my_conv = layers.Conv2D(C, OC, stride, padding="same", bias=True)
torch_conv = nn.Conv2d(C, OC, stride, padding=1, bias=True)
input = torch.Tensor(x)

weight_np = np.arange(OC*C*k_h*k_w)
weight_np = np.reshape(weight_np, (OC, C, k_h, k_w))
my_conv.weights["w"] = weight_np

weight_b = np.random.rand(OC)
my_conv.weights["b"] = weight_b

a = torch.Tensor(weight_np)
p = torch.nn.Parameter(a)
b = torch.Tensor(weight_b)
p2 = torch.nn.Parameter(b)

torch_conv.weight = p
torch_conv.bias = p2




y_torch = torch_conv(input)
# y_torch.backward(torch.Tensor(y))
print(y_torch)
print(my_conv(x))

#
# # 测试反向
# conv = layers.Conv2D(2, 1, 2, padding="same", use_bias=True)
# x = np.ones((2, 2, 2, 2), dtype=np.float32)
# x[1, ...] += 1
# y = np.random.rand(2, 1, 2, 2)
# y[1, :, :, :] += 1
# critterion = optims.Adam(conv)
# l = losses.MeanSquaredError()
#
# for i in range(10000):
#     y_pred = conv(x)
#     L = l(y_pred, y)
#     da = l.backwards()
#     conv.backwards(da)
#     critterion.step()
#
# # print(x)
# print(y)
# print("--------------------")

# print(conv.weights["w"])
# print(conv(x))