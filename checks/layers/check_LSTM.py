import numpy as np
from simpleNet import Moduel, layers
from simpleNet.utils.grad_check import grad_check

print("提供x dy")

class Net1(Moduel):
    def __init__(self):
        super().__init__()
        self.lstm = layers.LSTM(3, 5)

    def forwards(self, x):
        y, (_, _) = self.lstm(x)
        return y

    def backwards(self, da):
        dx = self.lstm.backwards(da)
        return dx

net = Net1()
x = np.random.rand(10, 4, 3)
grad_check(net, x)


print("")
print("提供x dh_last, ds_last")

class Net2(Moduel):
    def __init__(self):
        super().__init__()
        self.lstm = layers.LSTM(3, 5)
        self.concate = layers.Concate(axis=-1)

    def forwards(self, x):
        _, (dh, ds) = self.lstm(x)
        y = self.concate((dh, ds))
        return y

    def backwards(self, da):
        dh, ds = self.concate.backwards(da)
        dx = self.lstm.backwards(dhds=(dh, ds))
        return dx

net = Net2()
x = np.random.rand(10, 4, 3)
grad_check(net, x)

print("")
print("双层，第一层h0, s0接第一层")

class Net3(Moduel):
    def __init__(self):
        super().__init__()
        self.lstm1 = layers.LSTM(3, 5)
        self.lstm2 = layers.LSTM(4, 5)

    def forwards(self, x1, x2):
        _, (h, s) = self.lstm1(x1)
        y, (_, _) = self.lstm2(x2, (h, s))
        return y

    def backwards(self, da):
        dx2, (dh, ds) = self.lstm2.backwards(da)
        dx1, (dh0, ds0) = self.lstm1.backwards(dhds=(dh, ds))
        return dx1, (dh0, ds0)


net = Net3()
x1 = np.random.rand(2, 2, 3)
x2 = np.random.rand(2, 2, 4)
grad_check(net, (x1, x2))
