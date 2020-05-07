import numpy as np
from simpleNet import Moduel, layers
from simpleNet.utils.grad_check import grad_check

class Net(Moduel):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(3, 4)
        self.leakly_relu = layers.Leakly_Relu(0.2)

net = Net()
x = np.random.rand(10, 3)
grad_check(net, x)
