from simpleNet.layers.Layer import Layer
import numpy as np


class Embedding(Layer):
    """
        用向量表示ind
        [N, 1] -> [N, dims]
    """
    def __init__(self, num_embeddings:int, embedding_dim:int):
        super().__init__()
        self.name = "Embedding"
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weights["w"] = np.random.rand(num_embeddings, embedding_dim)

    def __call__(self, x):
        # x: [N, 1]
        x = x.astype(np.int32)
        out_dim = self.embedding_dim
        w = self.weights["w"]

        N = x.shape[0]
        y = np.zeros((N, out_dim))
        for i in range(N):
            y[i, :] = w[x[i], :]

        self.cached_x = x
        return y

    def backwards(self, da):
        # da: [N, D]
        in_dim, out_dim = self.num_embeddings, self.embedding_dim
        N, D = da.shape
        x = self.cached_x

        dw = np.zeros((in_dim, out_dim), dtype=da.dtype)

        for i in range(N):
            dw[x[i], ...] += da[i, ...]

        self.cached_grad = {"w": dw}
        return None

    def __str__(self):
        return "%s: in_dim: %d, out_dim: %d" % (self.name, self.num_embeddings, self.embedding_dim)
