import numpy as np
from simpleNet import layers, losses

loss = losses.BinaryCrossEntropy()

ep = 1e-12

y = 0
y_ = 0

y_true = np.array([[y]])
y_pred = np.array([[y_]])

l = loss(y_pred, y_true)
print(l)
da = loss.backwards()
print(da)


import torch
import torch.nn as nn


x = torch.Tensor([[y_]])
x.requires_grad = True
y = torch.Tensor([[y]])
y.requires_grad = False
loss = nn.BCELoss()
l = loss(x, y)
print(l)
l.backward()
print(x.grad)