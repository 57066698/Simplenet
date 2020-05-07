from simpleNet.layers.Layer import Layer
import numpy as np

"""
    Batch_Normalization
    
    --------------  
    在 [N, D] 上运行, 超过第二维在运算时，将参数化成[D, 1, 1, ....]
    
    mean[d] = X[d]  / N
    var[d] = (X[d] - mean[d]) ** 2  / N
    train_y[d] = gamma * (x[d] - mean[d]) / (var[d] + epsilon)**0.5 + beta
    
    running_mean = momentum * last_mean + (1 - momentum) * mean
    running_var = momentum * last_var + (1 - momentum) * var
    
    test_y[d] = gamma * (x[d] - running_mean[d]) / (running_var[d] + epsilon)**0.5 + beta
"""


class Batch_Normalization(Layer):
    def __init__(self, axis=1, momentum=0.99, epsilon=0.001):
        super().__init__()
        self.name = "Batch_Normalization"
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.inited = False

    def init_weights(self, x_shape):

        weights_shape = np.ones(len(x_shape), dtype=np.int)
        weights_shape[0] = x_shape[self.axis]  # [D, 1, 1, 1 ....]

        # 变换shape
        self.trans_shape = np.arange(len(x_shape))
        self.trans_shape[[0, self.axis]] = [self.trans_shape[self.axis], 0]  # 换回来也是用这个置换shape

        self.weights["gamma"] = np.ones(weights_shape)
        self.weights["beta"] = np.zeros(weights_shape)
        self.weights["running_mean"] = np.zeros(weights_shape)
        self.weights["running_var"] = np.ones(weights_shape)
        self.inited = True

    def __call__(self, x):
        """
            将axis 和 batch 互换
            运算
            然后换回去
        :param x:
        :return:
        """
        if not self.inited:
            self.init_weights(x.shape)

        momentum, epsilon = self.momentum, self.epsilon
        gamma, beta = self.weights['gamma'], self.weights['beta']
        running_mean, running_var = self.weights['running_mean'], self.weights['beta']
        trans_shape = self.trans_shape
        axis = self.axis
        sum_shape = tuple(range(len(x.shape)))[1:]

        assert x.shape[axis] == gamma.shape[0]

        # 获得除开axis外各轴的乘积
        muti = np.exp(np.sum(np.log(x.shape)) - np.log(x.shape[axis]))

        if self.mode == "train":
            # 1
            x_trans = np.transpose(x, trans_shape)
            # 2
            mean = np.sum(x_trans, axis=sum_shape, keepdims=True) / muti  # [D, 1, 1 ...]
            # 3
            xmean = x_trans - mean  # [D, N, 1, ...]
            # 4
            carre = xmean ** 2  # [D, N, 1, ...]
            # 5
            var = np.sum(carre, axis=sum_shape, keepdims=True) / muti  # [D, 1, 1 ...]
            # 6
            sqrtvar = np.sqrt(var + epsilon)  # [D, 1, 1 ...]
            # 7
            invvar = 1. / sqrtvar  # [D, 1, 1 ...]
            # 8
            x_hat = xmean * invvar  # [D, N, 1, ...]
            # 9
            y_trans = gamma * x_hat + beta  # [D, N, 1, ...]
            # 10
            y = np.transpose(y_trans, trans_shape)  # [N, D, 1, ...]

        elif self.mode == "test":
            x_trans = np.transpose(x, trans_shape)
            x_hat = (x_trans - running_mean) / np.sqrt(running_var + epsilon)
            y_trans = gamma * x_hat + beta
            y = np.transpose(y_trans, trans_shape)

        self.weights['running_mean'] = momentum * mean + (1 - momentum) * running_mean
        self.weights['running_var'] = momentum * var + (1 - momentum) * running_var
        self.caches = (gamma, beta, x_hat, invvar, xmean, sqrtvar, var, carre, mean, x_trans)

        return y

    def backwards(self, da):
        # da: [N, D]
        epsilon = self.epsilon
        trans_shape = self.trans_shape
        sum_shape = tuple(range(len(da.shape)))[1:]
        axis = self.axis
        gamma, beta, x_hat, invvar, xmean, sqrtvar, var, carre, mean, x_trans = self.caches

        assert da.shape[axis] == gamma.shape[0]
        assert len(da.shape) == len(x_trans.shape)

        # 获得除开axis外各轴的乘积
        muti = np.exp(np.sum(np.log(da.shape)) - np.log(da.shape[axis]))

        # 10
        dy_trans = np.transpose(da, trans_shape)  # [D, N, 1, ...]

        # 9
        dx_hat = dy_trans * gamma  # [D, N, 1, ...]
        dgamma = np.sum(dy_trans * x_hat, axis=sum_shape, keepdims=True)  # [D, 1, ...]
        dbeta = np.sum(dy_trans, axis=sum_shape, keepdims=True)  # [D, 1, ...]

        # 8
        dxmean = dx_hat * invvar  # [D, N, 1, ...]
        dinvvar = dx_hat * xmean  # [D, N, 1, ...]

        # 7
        dsqrtvar = -1. / (sqrtvar ** 2) * dinvvar  # [D, N, 1, ...]

        # 6
        dvar = 0.5 * (var + epsilon) ** (-0.5) * dsqrtvar  # [D, N, 1, ...]

        # 5
        dcarre = 1 / float(muti) * np.ones(carre.shape) * dvar  # [D, N, 1, ...]

        # 4
        dxmean += 2 * xmean * dcarre  # [D, N, 1, ...]

        # 3
        dx_trans = dxmean  # [D, N, 1, ...]
        dmean = - np.sum(dxmean, axis=sum_shape, keepdims=True)  # [D, 1, ...]

        # 2
        dx_trans += 1 / float(muti) * np.ones(x_trans.shape) * dmean  # [D, N, 1, ...]

        # 1
        dx = np.transpose(dx_trans, trans_shape)  # x shape

        self.cached_grad = {"gamma": dgamma, "beta": dbeta}

        return dx

    def __str__(self):
        return self.name
