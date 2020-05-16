import numpy as np
from simpleNet import Moduel, layers
from simpleNet.utils.grad_check import grad_check

conv = layers.Conv2D(3, 4, 2, 2, 'same', bias=True)
x = np.random.rand(10, 3, 5, 5)
grad_check(conv, x)
