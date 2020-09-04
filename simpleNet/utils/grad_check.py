import numpy as np
from simpleNet.layers.Layer import Layer
from simpleNet import Moduel


def grad_check(layer: Layer, inputs, start=0, loss=None, y_true=None):
    """
    梯度检查
    :param layer: 目标
    :param inputs: layer的输入, 单个或tuple
    :param start: 从第几个weight开始，顺序是深度优先
    :param check_inputs: 是否检查inputs，需要layer可以返回全部inputs的偏导
    :return:
    """
    epsilon = 1e-7



    if loss is None:
        loss = My_loss()

    X = []

    for i, x in enumerate(tuple2list(inputs)):
        X.append(x.astype(np.float64))
    X = tuple(X)

    change_all_weights(layer)
    y_pred = layer(*X)
    loss(y_pred, y_true)
    da = loss.backwards()
    layer.backwards(da)
    # layer.mode = 'test'  # 后面都是test了

    # weights 和 grads 的 list
    weight_dic = layer.weights
    grad_dic = layer.grads
    weight_list = _dic2list(weight_dic)
    grad_list = _dic2list(grad_dic)

    def get_grad(path, grad_list):
        for grad in grad_list:
            if grad['path'] == path:
                return grad['data']
        return None

    total_grad_1d = None
    total_gradapprox_1d = None

    # 逐个weight处理, 对每个weight取前100个
    for i in range(start, len(weight_list), 1):
        path = weight_list[i]['path']
        weight = weight_list[i]['data']
        grad = get_grad(path, grad_list)
        if grad is None:
            continue

        weight_num = np.prod(weight.shape)
        use_num = min(weight_num, 10)

        grads_1d = np.reshape(grad, (-1))[:use_num]
        gradapprox_1d = np.zeros(use_num)

        for j in range(use_num):
            ind = get_shape_ind(weight.shape, j)
            weight[ind] += epsilon
            J_plus = loss(layer(*X), y_true)
            weight[ind] -= 2 * epsilon
            J_minus = loss(layer(*X), y_true)
            weight[ind] += epsilon
            gradapprox_1d[j] = (J_plus - J_minus) / (2 * epsilon)

        if total_grad_1d is None:
            total_grad_1d = grads_1d
            total_gradapprox_1d = gradapprox_1d
        else:
            total_grad_1d = np.concatenate((total_grad_1d, grads_1d), axis=0)
            total_gradapprox_1d = np.concatenate((total_gradapprox_1d, gradapprox_1d), axis=0)
        numerator = np.linalg.norm(grads_1d - gradapprox_1d)
        denominator = np.linalg.norm(gradapprox_1d) + np.linalg.norm(grads_1d)
        diff = numerator / denominator

        print("")
        print("%d/%d %s: %.08f , shape %s ------------------" % (i, len(weight_list), path, diff, weight.shape))
        print(grads_1d)
        print(gradapprox_1d)

    numerator = np.linalg.norm(total_grad_1d - total_gradapprox_1d)
    denominator = np.linalg.norm(total_gradapprox_1d) + np.linalg.norm(total_grad_1d)
    diff = numerator / denominator

    print("")
    print("total: %.08f" % diff)


def tuple2list(t):
    if isinstance(t, tuple):
        lst = []
        for item in t:
            if isinstance(item, tuple):
                lst += tuple2list(item)
            else:
                lst.append(item)
        return lst
    else:
        return [t]


def change_all_weights(layer):
    if not isinstance(layer, Moduel):
        for key in layer.weights:
            layer.weights[key] = layer.weights[key].astype(np.float64)
    else:
        for l in layer.layers:
            change_all_weights(l)


def _dic2list(dic):
    """
    将weights dic 和 grad dic 转化为 list
    :param dic
    :return:
    """
    lst = []

    for key in dic:
        item = dic[key]
        if isinstance(item, dict):
            sub_list = _dic2list(item)
            for sub_item in sub_list:
                sub_item['path'] = key + "/" + sub_item['path']
            lst += sub_list
        else:
            lst.append({'path': key, 'data': item})
    return lst


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


class My_loss:

    def __call__(self, y_pred, y_true=None):
        """
            如果要令loss等于1，则输入 y_ture:   sum(y_pred - y_true) / N = 1
        """
        self.cached_y_pred = y_pred
        return np.sum(y_pred) / y_pred.shape[0]

    def backwards(self):
        """
            返回每个sample Loss 为1 的da
        """
        return np.ones(self.cached_y_pred.shape) / self.cached_y_pred.shape[0]
