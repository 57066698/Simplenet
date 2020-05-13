import numpy as np
from simpleNet.layers.Layer import Layer
from simpleNet import Moduel


def grad_check(layer: Layer, inputs, start=0):
    """
    梯度检查
    :param layer: 目标
    :param inputs: layer的输入, 单个或tuple
    :param start: 从第几个weight开始，顺序是深度优先
    :return:
    """
    epsilon = 1e-7

    X = []

    if isinstance(inputs, tuple):
        for x in inputs:
            X.append(x.astype(np.float64))
    else:
        X.append(inputs.astype(np.float64))
    X = tuple(X)

    change_all_weights(layer)
    y_pred = layer(*X)
    layer.backwards(loss_back(y_pred))

    N, weights, grads = fetch_all_weights(layer)

    grads_1d = np.array(grads).reshape((-1))
    gradapprox_1d = np.zeros(N)

    for i in range(N):
        change_by_total_index(weights, i, epsilon)
        J_plus = loss(layer(*X))
        change_by_total_index(weights, i, - 2 * epsilon)
        J_minus = loss(layer(*X))
        change_by_total_index(weights, i, epsilon)

        gradapprox_1d[i] = (J_plus - J_minus) / (2 * epsilon)

        if i % 1000 == 0:
            print(i)

    numerator = np.linalg.norm(grads_1d - gradapprox_1d)
    denominator = np.linalg.norm(gradapprox_1d) + np.linalg.norm(grads_1d)
    diff = numerator / denominator

    print(diff)

    print(grads_1d)
    print(gradapprox_1d)


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
        else:
            lst.append({'path':key, 'data':item})
    return lst

def fetch_all_weights(layer):
    N = 0
    weights = []
    grads = []

    if isinstance(layer, Moduel):
        for l in layer.layers:
            sub_N, sub_weights, sub_grads = fetch_all_weights(l)
            N += sub_N
            weights = weights + sub_weights
            grads = grads + sub_grads
    else:
        for key in layer.cached_grad:
            value = layer.cached_grad[key]
            grads = grads + np.reshape(value, (-1)).tolist()
            weights.append(layer.weights[key])
            N += np.reshape(value, (-1)).shape[0]

    return N, weights, grads


def change_by_total_index(lst_ndarray, index, value):
    """
    按照总index 序号， 修改lst_darray中的值
    :param lst_ndarray: [ndarray, ndarray, ...]
    :param index:
    :param value:
    :return:
    """

    def find_args(shape, index):
        """
        从 shape 中找到总第index个的的具体位置，没有就返回-1, 同时返回总数量
        :param shape:
        :param index:
        :return:
        """
        nums = []
        muti = 1
        for i in range(len(shape)-1, -1, -1):
            muti = muti * shape[i]
            nums.append(muti)
        nums = nums[::-1]
        nums.append(1)

        if index > nums[0] - 1:
            return -1, nums[0]

        inds = []
        remain = index
        for i in range(len(shape)):
            ind = remain//nums[i+1]
            remain = remain%nums[i+1]
            inds.append(ind)

        return inds, nums[0]

    checked_num = 0
    ndarray_ind = 0
    shape_ind = None
    for i in range(len(lst_ndarray)):
        shape = lst_ndarray[i].shape
        ind, num = find_args(shape, index-checked_num)
        if ind != -1:
            shape_ind = ind
            ndarray_ind = i
            break
        else:
            checked_num += num

    lst_ndarray[ndarray_ind][tuple(shape_ind)] += value

def loss(y_pred):
    return np.sum(y_pred) / y_pred.shape[0]
def loss_back(y_pred):
    return np.ones_like(y_pred) / y_pred.shape[0]
