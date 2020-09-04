import torch
import numpy as np


def loss(y_pred):
    return torch.sum(y_pred) / y_pred.shape[0]


def loss_backwards(y_pred):
    return torch.ones(y_pred.shape, dtype=torch.float64) / y_pred.shape[0]


def grad_check_torch_layer(torch_layer, x):
    epsilon = 1e-7

    torch_layer.double()
    x = x.double()

    y_pred = torch_layer(x)
    l = loss(y_pred)
    l.backward()

    # grads 和 weights
    weights = []
    grads = []
    weight_nums = []

    for key in torch_layer._parameters:
        weight = torch_layer._parameters[key]
        weights.append(weight)
        grads.append(weight.grad)
        weight_nums.append(np.prod(weight.shape, dtype=np.int))

    grad_1d_total = np.zeros(np.sum(weight_nums))
    gradapprox_1d_total = np.zeros(np.sum(weight_nums))

    for i in range(len(weights)):

        weight = weights[i]
        grad_1d = weight.grad.cpu().data.numpy().reshape(-1)
        gradapprox_1d = np.zeros(weight_nums[i])

        for j in range(weight_nums[i]):
            shape_ind = get_shape_ind(weight.shape, j)
            weight[shape_ind] += epsilon
            J_plus = loss(torch_layer(x)).cpu().data.numpy()
            weight[shape_ind] -= 2 * epsilon
            J_minus = loss(torch_layer(x)).cpu().data.numpy()
            weight[shape_ind] += epsilon
            gradapprox_1d[j] = (J_plus - J_minus) / (2 * epsilon)

        numerator = np.linalg.norm(grad_1d - gradapprox_1d)
        denominator = np.linalg.norm(gradapprox_1d) + np.linalg.norm(grad_1d)
        diff = numerator / denominator

        print("")
        print("%d/%d: %.08f , shape %s ------------------" % (i, len(weights), diff, weight.shape))
        print(grad_1d)
        print(gradapprox_1d)

        grad_1d_total[np.sum(weight_nums[:i], dtype=np.int): np.sum(weight_nums[:i + 1], dtype=np.int)] = grad_1d
        gradapprox_1d_total[np.sum(weight_nums[:i], dtype=np.int): np.sum(weight_nums[:i + 1], dtype=np.int)] = gradapprox_1d

    numerator = np.linalg.norm(grad_1d_total - gradapprox_1d_total)
    denominator = np.linalg.norm(gradapprox_1d_total) + np.linalg.norm(grad_1d_total)
    diff = numerator / denominator

    print("")
    print("total: %.08f" % diff)


def get_shape_ind(shape, n):
    """
    获得从shape中第n个index tuple
    :param shape: np shape
    :param n: int
    :return: (d1, d2, ... )
    """
    nums = []
    muti = 1
    for i in range(len(shape) - 1, -1, -1):
        muti = muti * shape[i]
        nums.append(muti)
    nums = nums[::-1]
    nums.append(1)

    if n > nums[0] - 1:
        return -1

    inds = []
    remain = n
    for i in range(len(shape)):
        ind = remain // nums[i + 1]
        remain = remain % nums[i + 1]
        inds.append(ind)

    return tuple(inds)
