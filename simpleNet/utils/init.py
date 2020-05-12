from simpleNet.layers.Layer import Layer
import numpy as np

"""
    一些weights初始化工具
    建好模型后用
    用法 simpleNet.utils.Normal_(weights, mean, std, key)  这样
    key 表示对包含key 的关键字使用
"""


def _select_weights(weights, key):
    selected_weights = []
    for k in weights:
        item = weights[k]
        if isinstance(item, dict):  # item 是 dict
            selected_weights += _select_weights(item, key)
        elif k.find(key) >= 0:  # item 是 nparray weight
            selected_weights.append(item)
    return selected_weights


def Normal_(weights: dict, mean: float, std: float, key: str = None):
    """
    正态分布
    :param weights: 目标weights
    :param mean: 均值
    :param std: 方差
    :param key: None 表示全部, "w" 表示各种变换矩阵, "b" 表示各种 bias, 以此类推
    :return:
    """
    selected_weights = _select_weights(weights, key)
    for i in range(len(selected_weights)):
        weight = selected_weights[i]
        weight[...] = np.random.normal(mean, std, weight.shape)


def Uniform_(weights: dict, low: float, high: float, key: str = None):
    """
    均匀分布 [low, high)
    :param weights:
    :param low:
    :param high:
    :param key: None 表示全部, "w" 表示各种变换矩阵, "b" 表示各种 bias, 以此类推
    :return:
    """
    selected_weights = _select_weights(weights, key)
    for i in range(len(selected_weights)):
        weight = selected_weights[i]
        weight[...] = np.random.uniform(low, high, weight.shape)
