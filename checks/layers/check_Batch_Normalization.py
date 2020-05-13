import numpy as np
from simpleNet import Moduel, layers
from simpleNet.utils.grad_check import grad_check, grad_check_one

bn = layers.Batch_Normalization(axis=1)
x = np.random.rand(10, 5, 4, 4)
grad_check_one(bn, x)
