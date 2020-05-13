"""
    RMSProp optimizer
"""
from simpleNet.layers import Layer
from simpleNet import Moduel
import numpy as np


class RMSProp:

    def __init__(self, model: Layer, lr: float = 1e-3, alpha: float = 0.99, eps: float = 1e-8):
        self.model = model
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.dic = {}
        self.inited = False

    def _zero_grad(self, layer):

        if isinstance(layer, Moduel):
            for sub_layer in layer.layers:
                self._zero_grad(sub_layer)
        else:
            self.dic[layer] = {}
            for key in layer.cached_grad:
                self.dic[layer][key] = {"s": None}

    def zero_grad(self):
        self._zero_grad(self.model)

    def _step(self, layer):
        if isinstance(layer, Moduel):
            for sub_layer in layer.layers:
                self._step(sub_layer)
        else:
            layer_grade_args = self.dic[layer]

            for key in layer.weights:
                args = layer_grade_args[key]
                grad = layer.cached_grad[key]
                s = args["s"]
                if s is None:
                    s = np.zeros(grad.shape)
                s = self.alpha * s + (1.0-self.alpha) * np.square(grad)
                args["s"] = s
                layer.weights[key] -= self.lr * grad / np.sqrt(s + self.eps)

    def step(self):
        if not self.inited:
            self.zero_grad()
            self.inited = True

        self._step(self.model)


