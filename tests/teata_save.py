import simpleNet
from simpleNet import layers, Moduel, losses, optims
import numpy as np


class block1(Moduel):
    def __init__(self):
        super().__init__()

        self.dense1 = layers.Dense(28 * 28, 512)
        self.relu1 = layers.Relu()
        self.dropout1 = layers.Dropout(0.2)

    def forwards(self, x):
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        return x

net = block1()
net.layers[0].weights["w"][0, 0] = 1

net.save_weights("../222.npz")



net2 = block1()
net2.load_weights("../222.npz")

print(net2.layers[0].weights["w"][0, 0])

