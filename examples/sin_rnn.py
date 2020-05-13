"""
    输入一个正弦函数，让rnn网络模拟对应的余弦
"""
import math
import numpy as np
from simpleNet import layers, losses, optims, Moduel
import matplotlib.pyplot as plt


def gen(N, length, rate):
    """
        生成 2*pi*length*rate个x, y
    """
    x = np.arange(length * rate) / rate * 2 * math.pi

    x = np.tile(np.expand_dims(x, 0), (N, 1))
    tint_x = np.random.rand(N) * 10
    # x = x + tint_x.T
    X = np.expand_dims(np.sin(x), -1)
    Y = np.expand_dims(np.cos(x), -1)

    return X, Y


def print_img(x, y):
    fig = plt.figure()
    ax = plt.axes()

    ax.plot(x, label='x')
    ax.plot(y, label="y")
    plt.axis('tight')
    plt.show()


class Net(Moduel):
    def __init__(self):
        super(Net, self).__init__()

        self.rnn = layers.Rnn(1, 10)
        self.dense = layers.Dense(10, 1)

    def forwards(self, x):
        x = self.rnn(x)
        x = self.dense(x)
        return x

net = Net()
loss = losses.MeanSquaredError()
optimizer = optims.Adam(net)

# train
for i in range(3000):
    X, Y = gen(10, 0.5, 20)
    y_pred = net(X)
    l = loss(y_pred, Y)
    if i % 100 == 0:
        print(l)
    da = loss.backwards()
    net.backwards(da)
    optimizer.step()

x, y = gen(1, 1, 20)
y_pred = net(x)
y_pred = y_pred.reshape(-1)
x = x.reshape(-1)
print_img(x, y_pred)