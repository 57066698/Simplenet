"""
    adam optimizer
    * 持有层
    * da 记录在各层
    * momentum, velocity 也记录在各层
"""
from simpleNet.layers import Layer
from simpleNet import Moduel


class Adam:
    def __init__(self, model:Layer, lr:float=0.001, fy1:float=0.9, fy2:float=0.99):
        self.model = model
        self.lr = lr
        self.fy1 = fy1
        self.fy2 = fy2
        self.num_step = 0
        self.hom = 10e-8
        self.inited = False
        self.dic = {}

    def _zero_grad(self, layer):

        if isinstance(layer, Moduel):
            for sub_layer in layer.layers:
                self._zero_grad(sub_layer)
        else:
            self.dic[layer] = {}
            for key in layer.cached_grad:
                self.dic[layer][key] = {"s": 0, "r": 0}

    def zero_grad(self):
        self._zero_grad(self.model)
        self.num_step = 1

    def _step(self, layer):
        if isinstance(layer, Moduel):
            for sub_layer in layer.layers:
                self._step(sub_layer)
        else:
            layer_grade_args = self.dic[layer]

            for key in layer.cached_grad:
                grad = layer.cached_grad[key]
                args = layer_grade_args[key]

                args["s"] = self.fy1 * args["s"] + (1 - self.fy1) * grad
                args["r"] = self.fy2 * args["r"] + (1 - self.fy2) * grad * grad

                s_hat = args["s"] / (1 - self.fy1 ** self.num_step)
                r_hat = args["r"] / (1 - self.fy2 ** self.num_step)
                theta = self.lr * (s_hat / (r_hat ** 0.5 + self.hom))
                layer.weights[key] -= theta

    def step(self):

        if not self.inited:
            self.zero_grad()
            self.inited = True

        self._step(self.model)

