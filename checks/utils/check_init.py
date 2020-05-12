import numpy as np
from simpleNet import layers, Moduel, init

class Net(Moduel):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(2, 3)
        self.block = Moduel([layers.Dense(3, 1)])

net = Net()
weights = net.weights
print(weights)

init.Normal_(weights, 0, 0.2, key="w")
print(weights)