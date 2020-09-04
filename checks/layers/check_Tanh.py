import numpy as np
from simpleNet import Moduel, layers
from simpleNet.utils.grad_check import grad_check


bn = Moduel([layers.Dense(2, 3), layers.Sigmoid()])
x = np.random.rand(5, 2)
grad_check(bn, x)