import numpy as np
from simpleNet import layers
from torch import nn, optim
import torch
from torch.autograd import gradcheck


x = np.random.rand(10, 2, 3, 3).astype(np.float64)
N = x.shape[0]
epsilon = 1e-7


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense = nn.Conv2d(2, 3, 2, bias=False)

    def forward(self, x):
        x = self.dense(x)
        return x

net = Net()
net.double()

critterion = nn.CrossEntropyLoss()

x1 = torch.Tensor(x).double()
x1.requires_grad=True

print(gradcheck(net, x1))