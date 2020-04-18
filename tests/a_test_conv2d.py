# 测试 Conv2d
import numpy as np
from simpleNet import layers, optims, losses
import torch
from torch import nn

# 测试正向

x = np.arange(2*3*5*5)
x = np.reshape(x, (2, 3, 5, 5))

my_conv = layers.Conv2D(3, 2, 3, padding="same", use_bias=True)
torch_conv = nn.Conv2d(3, 2, 3, padding=1, bias=True)
input = torch.Tensor(x)

weight_np = np.random.rand(2, 3, 3, 3)
my_conv.weights[0] = weight_np

weight_b = np.random.rand(2)
my_conv.weights[1] = weight_b

with torch.no_grad():
    a = torch.Tensor(weight_np)
    p = torch.nn.Parameter(a)
    b = torch.Tensor(weight_b)
    p2 = torch.nn.Parameter(b)

    torch_conv.weight = p
    torch_conv.bias = p2

print(torch_conv(input).size())
print(torch_conv(input)[0, 0, 1, 1].data.numpy())
print(my_conv(x)[0, 0, 1, 1])

# 测试反向
conv = layers.Conv2D(1, 1, 2, padding="same", use_bias=False)
x = np.ones((1, 1, 2, 2))
y = np.array([[[[4, 2],[2, 1]]]])
critterion = optims.Adam(conv)
l = losses.MeanSquaredError()

for i in range(10000):
    y_pred = conv(x)
    L = l(y_pred, y)
    da = l.backwards()
    conv.backwards(da)
    critterion.step()

print(x)
print(y[0, 0, :, :])
print("--------------------")

print(conv.weights[0])
print(conv(x)[0, 0, :, :])