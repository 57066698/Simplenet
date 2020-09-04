import simpleNet
from simpleNet import layers, Moduel, losses, optims
from examples.datasets.mnist_loader import load_train, load_test
import numpy as np

X_train, Y_train = load_train()
X_test, Y_test = load_test()

print(X_train.shape, Y_train.shape)


class Net(Moduel):
    # forward网络, (512 Dense 0.2 Dropout) x 2 -> softmax -> crossentropy
    def __init__(self):
        super().__init__()

        self.conv1 = layers.Conv2D(1, 16, 3, padding='same', bias=False)
        self.bn1 = layers.Batch_Normalization(axis=1)
        self.relu1 = layers.Relu()
        self.pool1 = layers.MaxPooling2D(2)

        self.conv2 = layers.Conv2D(16, 32, 3, padding='same', bias=False)
        self.bn2 = layers.Batch_Normalization(axis=1)
        self.relu2 = layers.Relu()
        self.pool2 = layers.MaxPooling2D(2)

        self.dropout1 = layers.Dropout(0.25)
        self.flatten = layers.Flatten()

        self.dense1 = layers.Dense(32*7*7, 128)
        self.relu3 = layers.Relu()
        self.dropout2 = layers.Dropout(0.5)

        self.dense2 = layers.Dense(128, 10)

# 数据gen
class Gen:
    def __init__(self, X, Y, batch_size:int = 32):
        self.batch_size = batch_size
        X = np.transpose(X, (0, 3, 1, 2))
        self.X = X / 255.0
        self.Y = np.zeros((Y.shape[0], 10), dtype=np.float32)

        for i in range(Y.shape[0]):
            self.Y[i][Y[i]] = 1
        self.inds = np.arange(X.shape[0])
        self.end_epoch()

    def next_batch(self, ind:int):
        left = ind * self.batch_size
        right = (ind+1) * self.batch_size
        right = min(right, self.inds.shape[0])

        batch_inds = self.inds[left:right]

        batch_X = self.X[batch_inds]
        batch_Y = self.Y[batch_inds]

        return batch_X, batch_Y

    def __len__(self):
        import math
        return math.ceil(self.inds.shape[0] / self.batch_size)

    def end_epoch(self):
        np.random.shuffle(self.inds)

    def totol_num(self):
        return self.inds.shape[0]


# train

net = Net()
net.summary()

criterrion = losses.SoftmaxCrossEntropy()
optimizer = optims.Adam(net)
gen_train = Gen(X_train, Y_train)
gen_test = Gen(X_train, Y_train)

epochs = 20
batch_size = 32

for i in range(epochs):
    print("-------------- epochs %d --------------------" %i)
    for j in range(len(gen_train)):
        X, Y = gen_train.next_batch(j)
        netout = net(X)
        loss = criterrion(netout, Y)
        if j % 10 == 0:
            print("train batch %d losses:" % j, loss)
        da = criterrion.backwards()
        net.backwards(da)
        optimizer.step()
    gen_train.end_epoch()

    # val
    num_right = 0
    num_total = gen_test.totol_num()
    for k in range(len(gen_test)):
        X, Y = gen_test.next_batch(k)
        netout = net(X, run=True)
        Y_pred = np.expand_dims(np.argmax(netout, axis=1), axis=-1)
        Y_pred = np.squeeze(Y_pred)
        y_list = np.argmax(Y, axis=-1)
        right = Y_pred == y_list
        num_right += np.sum(right.astype(np.int))
    print("accuracy: ", num_right / num_total)


