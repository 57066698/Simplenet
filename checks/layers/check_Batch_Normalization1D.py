import numpy as np
from simpleNet import Moduel, layers
from simpleNet.utils.grad_check import grad_check


class net(Moduel):
    def __init__(self):
        super().__init__()

        self.conv1 = layers.Dense(1, 1)
        self.bn1 = layers.Batch_Normalization1D(1)
        self.dense1 = layers.Dense(1, 1)
        self.bn2 = layers.Batch_Normalization1D(1)


bn = net()
x = np.random.rand(5, 1)
grad_check(bn, x)