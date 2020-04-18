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

    def zero_grad(self):

        def zero_layer(layer):
            self.dic[layer] = []
            for i in range(len(layer.cached_grad)):
                self.dic[layer].append({"s": 0, "r": 0})

        def zero_model(model):
            for layer in model.layers:
                if isinstance(layer, Moduel):
                    zero_model(layer)
                else:
                    zero_layer(layer)

        if isinstance(self.model, Moduel):
            zero_model(self.model)
        else:
            zero_layer(self.model)
        self.num_step = 1

    def step(self):

        if not self.inited:
            self.zero_grad()
            self.inited = True

        self.num_step += 1

        def step_layer(layer):
            grad_args = self.dic[layer]

            for i in range(len(layer.cached_grad)):
                grad = layer.cached_grad[i]

                s = grad_args[i]["s"]
                r = grad_args[i]["r"]

                s = self.fy1 * s + (1 - self.fy1) * grad
                r = self.fy2 * r + (1 - self.fy2) * grad * grad

                grad_args[i]["s"] = s
                grad_args[i]["r"] = r

                s_hat = s / (1-self.fy1**self.num_step)
                r_hat = r / (1-self.fy2**self.num_step)
                theta = - self.lr * (s_hat / (r_hat ** 0.5 + self.hom))
                layer.weights[i] += theta

        def step_moduel(moduel):

            for layer in moduel.layers:
                if isinstance(layer, Moduel):
                    step_moduel(layer)
                else:
                    step_layer(layer)

        if isinstance(self.model, Moduel):
            step_moduel(self.model)
        else:
            step_layer(self.model)

