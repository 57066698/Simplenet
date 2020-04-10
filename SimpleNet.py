import simpleNet
from simpleNet.MeanSqauredError import MeanSquaredError
from simpleNet.Moduel import Moduel
import numpy as np
from simpleNet.optim import Adam

x = np.random.rand(4, 5)
y = np.array([6, 0, 2, 1])
y = y.reshape(y.shape[0], 1)


class Model(Moduel):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = simpleNet.Dense(5, 3, use_bias=True)
        self.dense2 = simpleNet.Dense(3, 1, use_bias=True)

    def forwards(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x

model = Model()
criterion = MeanSquaredError()
optim = Adam(model)

for i in range(10000):
    y_pred = model.forwards(x)
    loss = criterion(y_pred, y)
    da = criterion.backwards()
    model.backwards(da)
    optim.step()

print(model.forwards(x))