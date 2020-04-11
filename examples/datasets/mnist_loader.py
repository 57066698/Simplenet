import gzip
import os
import numpy as np


def load_train():
    return _load('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')


def load_test():
    return _load('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')


def _load(img_file_name, label_file_name):

    f = gzip.open(os.path.dirname(__file__) + "/" + img_file_name, 'r')

    image_size = 28

    f.read(16)
    buf = f.read()
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    X = data.reshape(-1, image_size, image_size, 1)

    f = gzip.open(os.path.dirname(__file__) + "/" + label_file_name, 'r')
    f.read(8)
    buf = f.read()
    Y = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return X, Y
