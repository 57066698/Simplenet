import numpy as np
from simpleNet import Moduel, layers
from simpleNet.utils.grad_check import grad_check

bn = layers.Embedding(3, 5)
x = np.random.rand(10, 1) * 3
x = x.astype(np.int)
print(x)
grad_check(bn, x)
print(bn(x))
