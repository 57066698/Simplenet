import numpy as np
from simpleNet import Moduel, layers
from simpleNet.utils.grad_check import grad_check


bn = layers.Batch_Normalization2D(3, 4)
x = np.random.rand(2, 3, 2, 2)
grad_check(bn, x, start=0)