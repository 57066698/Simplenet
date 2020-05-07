import numpy as np
from simpleNet import layers, Moduel, optims, losses
from examples.datasets.mnist_loader import load_train, load_test
import matplotlib.pyplot as plt
from PIL import Image

# net
class Encoder(Moduel):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(1, 16, 3, stride=2, padding="same")
        self.relu1 = layers.Relu()
        self.conv2 = layers.Conv2D(16, 32, 3, stride=2, padding="same")
        self.relu2 = layers.Relu()
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(32*7*7, 16)

    def forwards(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


class Decoder(Moduel):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(16, 32 * 7 * 7)
        self.reshape = layers.Reshape((-1, 32, 7, 7))
        self.deconv1 = layers.Conv2DTranspose(32, 32, 3, stride=2)
        self.relu1 = layers.Relu()
        self.deconv2 = layers.Conv2DTranspose(32, 16, 3, stride=2)
        self.relu2 = layers.Relu()
        self.deconv3 = layers.Conv2DTranspose(16, 1, 3, stride=1)
        self.sigmoid = layers.Sigmoid()

    def forwards(self, x):
        x = self.dense(x)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.relu2(x)
        x = self.deconv3(x)
        x = self.sigmoid(x)
        return x


class AutoEncoder(Moduel):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forwards(self, x):
        latent = self.encoder(x)
        y = self.decoder(latent)
        return y

# dataset and gen

X_train, Y_train = load_train()
X_test, Y_test = load_test()

print(X_train.shape, Y_train.shape)


# 数据gen
class Gen:
    def __init__(self, X, batch_size:int = 32):
        self.batch_size = batch_size
        X = np.transpose(X, (0, 3, 1, 2))
        X = X / 255.0
        self.X = X
        self.inds = np.arange(X.shape[0])
        self.end_epoch()

    def next_batch(self, ind:int):
        left = ind * self.batch_size
        right = (ind+1) * self.batch_size
        right = min(right, self.inds.shape[0])

        batch_inds = self.inds[left:right]

        X = self.X[batch_inds]
        noise = np.random.normal(loc=0.5, scale=0.5, size=X.shape)
        noise_X = noise + X
        noise_X = np.clip(noise_X, 0, 1.)

        return noise_X, X

    def __len__(self):
        import math
        return math.ceil(self.inds.shape[0] / self.batch_size)

    def end_epoch(self):
        np.random.shuffle(self.inds)

    def totol_num(self):
        return self.inds.shape[0]


# printer
def show_img(imgs, epochs, rows=10, cols=10, image_size=28):
    imgs = imgs.reshape((rows*3, cols, image_size, image_size))
    imgs = np.vstack(np.split(imgs, rows, axis=1))
    imgs = imgs.reshape((rows * 3, -1, image_size, image_size))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    imgs = (imgs * 255).astype(np.uint8)
    plt.figure()
    plt.axis('off')
    plt.title('Original images: top rows, '
              'Corrupted Input: middle rows, '
              'Denoised Input:  third rows')
    plt.imshow(imgs, interpolation='none', cmap='gray')
    Image.fromarray(imgs).save('corrupted_and_denoised_%d.png' %epochs)


# train
autoencoder = AutoEncoder()
autoencoder.summary()
optim = optims.Adam(autoencoder)
loss = losses.MeanSquaredError()

gen_train = Gen(X_train, batch_size=32)
gen_test = Gen(X_test, batch_size=100)

autoencoder.load_weights("111.npz")

for i in range(30):
    print("-------------- epochs %d --------------------" % i)
    for j in range(len(gen_train)):
        X, Y = gen_train.next_batch(j)
        pred_Y = autoencoder(X)
        l = loss(pred_Y, Y)
        da = loss.backwards()
        autoencoder.backwards(da)
        optim.step()
        if j % 10 == 0:
            print("train batch %d losses:" % j, l)

    gen_train.end_epoch()
    autoencoder.save_weights("111.npz")
    # show test imgs
    rows, cols = 10, 10
    noise_img, img = gen_test.next_batch(0)
    netout = autoencoder(img, run=True)
    imgs = np.concatenate([img[:rows*cols], noise_img[:rows*cols], netout[:rows*cols]])
    show_img(imgs, i)
