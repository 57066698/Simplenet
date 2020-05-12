import numpy as np
from simpleNet import layers, Moduel, optims, losses, init
import matplotlib.pyplot as plt
"""
    AC_Gan 论文实现
"""

# model
class Generator(Moduel):
    def __init__(self, latent_dim, n_classes=10):
        super().__init__()
        # x: [N, latent_dim + n_classes]

        self.in_latent = Moduel([
            layers.Dense(latent_dim, 128 * 7 * 7),
            layers.Relu(),
            layers.Reshape((-1, 128, 7, 7))
        ])

        self.in_class = Moduel([
            layers.Embedding(n_classes, 50),
            layers.Dense(50, 7 * 7),
            layers.Reshape((-1, 1, 7, 7))
        ])

        self.concate = layers.Concate(axis=1)  # [N, 129, 7, 7]

        self.convTrans1 = Moduel([
            layers.Conv2DTranspose(129, 64, 5, 2, padding="same", bias=False),
            layers.Batch_Normalization(),
            layers.Relu()
        ])

        self.convTrans2 = Moduel([
            layers.Conv2DTranspose(64, 1, 5, 2, padding='same', bias=False),
            layers.Tanh()
        ])

    def forwards(self, latent_batch, class_ind_batch):
        x1 = self.in_latent(latent_batch)
        x2 = self.in_class(class_ind_batch)

        x = self.concate((x1, x2))

        x = self.convTrans1(x)
        x = self.convTrans2(x)

        return x

    def backwards(self, da):
        dx = self.convTrans2.backwards(da)
        dx = self.convTrans1.backwards(dx)

        dx1, dx2 = self.concate.backwards(dx)

        self.in_class.backwards(dx2)
        self.in_latent.backwards(dx1)

        return None


class Discriminator(Moduel):

    def conv_block(self, in_channel, out_channel, kernel, stride, bn, dp):

        block = Moduel()
        block.conv = layers.Conv2D(in_channel, out_channel, kernel, stride, padding='same', bias=False)
        if bn:
            block.bn = layers.Batch_Normalization()
        block.leaklyRelu = layers.Leakly_Relu(0.2)
        if dp:
            block.dropout = layers.Dropout(0.5)
        return block

    def __init__(self):
        # x [N, 1, 32, 32]
        super().__init__()

        self.convs = Moduel([
            self.conv_block(1, 16, 3, 2, bn=False, dp=False),
            self.conv_block(16, 32, 3, 1, bn=True, dp=True),
            self.conv_block(32, 64, 3, 2, bn=True, dp=True),
            self.conv_block(64, 128, 3, 1, bn=True, dp=True),
        ])

        # x [N, 128, 7, 7]
        self.flatten = layers.Flatten()

        # x [N, 128 * 7 * 7]
        self.out_1 = Moduel([
            layers.Dense(128 * 7 * 7, 1),
            layers.Sigmoid()
        ])

        self.out2 = Moduel([
            layers.Dense(128 * 7 * 7, 10),
            layers.Softmax()
        ])

    def forwards(self, x):
        x = self.convs(x)
        x = self.flatten(x)
        ture_false = self.out_1(x)
        clas = self.out2(x)
        return ture_false, clas

    def backwards(self, dture_false, dclas):
        dture_false = self.out_1.backwards(dture_false)
        # dclas = self.out2.backwards(dclas)
        dx = dture_false  # + dclas
        dx = self.flatten.backwards(dx)
        dx = self.convs.backwards(dx)
        return dx


# data and gen
# db = np.load("./datasets/fmnist.npz")
# x_train, y_train, x_test, y_test = db["x_train"], db["y_train"], db["x_test"], db["y_test"]

from examples.datasets.mnist_loader import load_train, load_test

x_train, y_train = load_train()


# 数据gen
class RealGen:
    def __init__(self, X, Y, batch_size: int = 64):
        self.batch_size = batch_size
        X = np.reshape(X, (X.shape[0], 1, 28, 28))
        self.X = X / 255.0
        self.Y = Y
        self.inds = np.arange(X.shape[0])
        self.end_epoch()

    def next_batch(self, ind: int):
        left = ind * self.batch_size
        right = (ind + 1) * self.batch_size
        right = min(right, self.inds.shape[0])

        batch_inds = self.inds[left:right]

        batch_X = self.X[batch_inds]
        batch_Y = self.Y[batch_inds]

        return batch_X, batch_Y

    def __len__(self):
        import math
        return math.floor(self.inds.shape[0] / self.batch_size)  # 避免少出

    def end_epoch(self):
        np.random.shuffle(self.inds)

    def totol_num(self):
        return self.inds.shape[0]


def one_hot(x):
    x = x.astype(np.int)
    onehot = np.zeros((x.size, 10))
    onehot[np.arange(x.size), x] = 1
    return onehot


# show
def show_imgs(imgs, batch=0):
    N = min(imgs.shape[0], 100)
    imgs *= 255.0
    imgs = np.transpose(imgs, (0, 2, 3, 1))

    for i in range(N):
        plt.subplot(10, 10, 1 + i)
        plt.axis('off')
        plt.imshow(imgs[i, :, :, 0], cmap='gray_r')

    filename = 'ac_gan_batch_%d.png' % batch
    plt.savefig(filename)
    plt.close()


# train
latent_dim = 100
batch_size = 128
n_class = 10

generator = Generator(latent_dim, n_class)
discriminator = Discriminator()
init.Normal_(generator, 0, 0.2, "w")  # 对所有名字含w的weight重新初始化
init.Normal_(discriminator, 0, 0.2, "w")
generator.summary()
discriminator.summary()
half_batch = int(batch_size / 2)
real_gen = RealGen(x_train, y_train, half_batch)
loss_real = losses.BinaryCrossEntropy()
loss_cls = losses.CategoricalCrossEntropy()
optim_g = optims.Adam(generator, lr=0.0002, fy1=0.5)
optim_d = optims.Adam(discriminator, lr=0.0002, fy1=0.5)


def latent_gen(batch_size, latent_dim=100, n_class=10):
    latent = np.random.randn(batch_size, latent_dim)
    clas = np.random.randint(0, n_class, batch_size)
    return latent, clas


def train_discriminator(x_imgs, y_real, y_cls):
    y_real_pred, y_cls_pred = discriminator(x_imgs)
    l_real, l_cls = loss_real(y_real_pred, y_real), loss_cls(y_cls_pred, y_cls)
    da_real, da_cls = loss_real.backwards(), loss_cls.backwards()
    discriminator.backwards(da_real, da_cls)
    optim_d.step()
    return l_real, l_cls


def train_generator(latent, y_real, y_cls):
    x_imgs = generator(latent, y_cls)
    y_real_pred, y_cls_pred = discriminator(x_imgs)
    l_real, l_cls = loss_real(y_real_pred, y_real), loss_cls(y_cls_pred, one_hot(y_cls))
    da_real, da_cls = loss_real.backwards(), loss_cls.backwards()
    dimg = discriminator.backwards(da_real, da_cls)
    generator.backwards(dimg)
    optim_g.step()
    return l_real, l_cls


for i in range(10):
    for j in range(len(real_gen)):
        # 真的一半，训D
        x_imgs, y_cls = real_gen.next_batch(j)  # [N/2, 1, 28, 28], [N/2, ]
        y_real = np.ones((half_batch, 1), dtype=np.float)
        l_s1, l_c1 = train_discriminator(x_imgs, y_real, one_hot(y_cls))

        # 假的一半，训D
        latent, y_cls = latent_gen(half_batch)
        x_imgs = generator(latent, y_cls)
        y_real = np.zeros((half_batch, 1), dtype=np.float)
        l_s2, l_c2 = train_discriminator(x_imgs, y_real, one_hot(y_cls))

        # 假的训G
        latent, y_cls = latent_gen(batch_size)
        y_real = np.ones((batch_size, 1), dtype=np.float)
        l_s3, l_c3 = train_generator(latent, y_real, y_cls)

        # print(l_s1, l_c1)

        print("%d/%d D_true: [l_s %.3f, l_c %.3f], D_fake: [l_s %.3f, l_c %.3f] G: [l_s %.3f, l_c %.3f]"
              % (j, len(real_gen), l_s1, l_c1, l_s2, l_c2, l_s3, l_c3))

    print("-------------------")
    latent, y_cls = latent_gen(half_batch)
    x_imgs = generator(latent, y_cls)
    show_imgs(x_imgs, i)
    generator.save_weights("gen.npz")
    discriminator.save_weights("dis.npz")
