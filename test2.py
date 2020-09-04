import torch
import torch.nn as nn
from simpleNet.utils.grad_check_torch import grad_check_torch_layer


bn = nn.BatchNorm2d(3).double()
x = torch.rand((2, 3, 4, 4), requires_grad=True).double()

grad_check_torch_layer(bn, x)
