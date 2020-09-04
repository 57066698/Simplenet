from simpleNet.layers.Layer import Layer
import numpy as np

"""
    Batch_Normalization
    
    --------------  
    1D
    
    mean[d] = X[d]  / N
    var[d] = (X[d] - mean[d]) ** 2  / N
    x_hat[d] = (x[d] - mean[d]) / (var[d] + epsilon)**0.5
    train_y[d] = gamma[d] * x_hat[d] + beta[d]
    
    running_mean = momentum * last_mean + (1 - momentum) * mean
    running_var = momentum * last_var + (1 - momentum) * var
    
    test_y[d] = gamma * (x[d] - running_mean[d]) / (running_var[d] + epsilon)**0.5 + beta
"""


class Batch_Normalization1D(Layer):
    def __init__(self, in_channels, momentum=0.99, epsilon=1e-12):
        """
        将batch 和 非 C 通道的数据空间拉平
        :param axis: C 通道位置
        :param momentum:
        :param epsilon:
        """
        super().__init__()
        self.name = "Batch_Normalization"
        self.momentum = momentum
        self.epsilon = epsilon
        self.in_channels = in_channels
        self.weights["gamma"] = np.ones((1, in_channels))
        self.weights["beta"] = np.zeros((1, in_channels))
        self.weights["running_mean"] = np.zeros((1, in_channels))
        self.weights["running_var"] = np.ones((1, in_channels))



    def __call__(self, x):

        momentum, epsilon = self.momentum, self.epsilon
        gamma, beta = self.weights['gamma'], self.weights['beta']
        running_mean, running_var = self.weights['running_mean'], self.weights['beta']

        assert len(x.shape) == 2 and x.shape[1] == self.in_channels
        N = x.shape[0]

        if self.mode == "train":
            # 1
            mean = np.sum(x, axis=0, keepdims=True) / N  # [1, D]
            # 2
            xmean = x - mean  # [N, D]
            # 3
            xmean_square = xmean ** 2  # [N, D]
            # 4
            var = np.sum(xmean_square, axis=0, keepdims=True) / N  # [1, D ...]
            # 5
            sqrtvar = np.sqrt(var + epsilon)  # [1, D]
            # 6
            invvar = 1. / sqrtvar  # [1, D]
            # 7
            x_hat = xmean * invvar  # [N, D]
            # 8
            y = gamma * x_hat + beta  # [N, D]

            self.weights['running_mean'] = momentum * mean + (1 - momentum) * running_mean
            self.weights['running_var'] = momentum * var + (1 - momentum) * running_var
            self.caches = (gamma, beta, x_hat, invvar, xmean, sqrtvar, var, xmean_square, mean)

        elif self.mode == "test":
            x_hat = (x - running_mean) / np.sqrt(running_var + epsilon)
            y = gamma * x_hat + beta

        return y

    def backwards(self, da):
        # da: [N, D]
        epsilon = self.epsilon
        in_channels = self.in_channels
        gamma, beta, x_hat, invvar, xmean, sqrtvar, var, xmean_square, mean = self.caches

        assert len(da.shape) == 2 and da.shape[1] == self.in_channels
        N = da.shape[0]

        # 8
        dx_hat = da * gamma  # [N, D]
        dgamma = np.sum(da * x_hat, axis=0, keepdims=True)  # [1, D]
        dbeta = np.sum(da, axis=0, keepdims=True)  # [1, D]

        # 7
        dxmean1 = dx_hat * invvar  # [N, D] * [1, D] = [N, D]
        dinvvar = np.sum(dx_hat * xmean, axis=0, keepdims=True)  # [1, D]

        # 6
        dsqrtvar = -1. / (sqrtvar ** 2) * dinvvar  # [1, D]

        # 5
        dvar = 0.5 * (var + epsilon) ** (-0.5) * dsqrtvar  # [1, D]

        # 4
        dxmean_square = np.ones((N, in_channels)) * dvar / N  # [N, D]

        # 3
        dxmean2 = 2 * xmean * dxmean_square  # [N, D]

        # 2
        dx1 = dxmean1 + dxmean2  # [N, D]
        dmean = -1.0 * np.sum(dxmean1+dxmean2, axis=0, keepdims=True)  # [1, D]

        # 1
        dx2 = np.ones((N, in_channels)) * dmean / N  # [N, D]
        dx = dx1 + dx2

        self.cached_grad = {"gamma": dgamma, "beta": dbeta}

        return dx

    def __str__(self):
        return self.name
