"""
    adam optimizer
    * 持有层
    * da 记录在各层
    * momentum, velocity 也记录在各层
"""
from simpleNet.layers import Layer
from simpleNet import Moduel


class SGD:

    def __init__(self, model: Layer, lr: float=0.0001):
        self.model = model
        self.lr = lr

    def _step(self, layer):
        if isinstance(layer, Moduel):
            for sub_layer in layer.layers:
                self._step(sub_layer)
        else:
            for key in layer.weights:
                grad = layer.cached_grad[key]
                layer.weights[key] -= self.lr * grad

    def step(self):
        self._step(self.model)


