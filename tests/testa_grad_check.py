import numpy as np
from simpleNet import layers
from simpleNet.utils.grad_check import grad_check
from simpleNet.layers.other_conv2d import Conv2d_other

# conv = Conv2d_other(2, 3, 2)
conv = layers.Conv2D(2, 3, 2, padding="valid", bias=True)
x = np.random.rand(10, 2, 2, 2)
epsilon = 1e-7

grad_check(conv, x)

dense = layers.Dense(3, 5)
x = np.random.rand(3, 3)
grad_check(dense, x)