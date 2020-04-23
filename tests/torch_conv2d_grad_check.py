import numpy as np
from simpleNet import layers
from torch import nn
import torch
from torch.autograd import gradcheck


conv = nn.Conv2d(2, 3, 2, padding=0, bias=False)
conv.double()
x = np.random.rand(10, 2, 2, 2).astype(np.float64)
y = np.random.rand(10, 3, 1, 1).astype(np.float64)
N = x.shape[0]
epsilon = 1e-7

# x1 = torch.Tensor(x)
# x1.requires_grad=True
# gradcheck(conv, x1)






def relu(z):
    return torch.max(z, torch.zeros_like(z))

def relu_back(a, da):
    z = da.clone()
    z[a<=0] = 0
    return z

def binary_cross_entropy(y_pred, y):
    return 0.5 * torch.sum(torch.pow((y_pred - y), 2)) / y_pred.shape[0]


def forward(x, y):
    x = torch.Tensor(x).double()
    x.requires_grad = True
    y = torch.Tensor(y).double()
    z = conv(x)
    a = relu(z)
    l = binary_cross_entropy(a, y)
    return l


# grad
l = forward(x, y)
l.backward()

grad = conv.weight.grad


# 计算gradapprox
w = grad.clone()
gradapprox = torch.zeros_like(w)

OC, IC, W, H = w.shape

for i in range(OC):
    for j in range(IC):
        for w in range(W):
            for h in range(H):
                # weight + - ，再算 J- J+
                conv.weight[i, j, w, h] += epsilon
                J_plus = forward(x, y)
                conv.weight[i, j, w, h] -= 2*epsilon
                J_minus = forward(x, y)

                gradapprox[i][j] = (J_plus - J_minus) / (2 * epsilon)
                conv.weight[i, j, w, h] += epsilon

# gradapprox /= 12
# 结果
grad = grad.data.numpy()
gradapprox = gradapprox.data.numpy()

numerator = np.linalg.norm(grad - gradapprox)
denominator = np.linalg.norm(gradapprox) + np.linalg.norm(grad)
diff = numerator / denominator

print(diff)