from simpleNet import layers, Moduel, optims, losses, init
import numpy as np
from simpleNet.utils.grad_check import grad_check

convTrans = Moduel([
            # layers.Dense(129, 64),
            layers.Conv2DTranspose(129, 64, 5, 2, padding="same"),
            # layers.Batch_Normalization2D(64),
        ])


x = np.random.rand(2, 129, 10, 10)

grad_check(convTrans, x)