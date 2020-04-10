"""
    adam optimizer
    * 持有层
    * da 记录在各层
    * momentum, velocity 也记录在各层
"""
from simpleNet.layers import Layer
from simpleNet import Moduel

class Adam:
    def __init__(self, model:Moduel, lr:float=0.001, fy1:float=0.9, fy2:float=0.99):
        self.model = model
        self.lr = lr
        self.fy1 = fy1
        self.fy2 = fy2
        self.num_step = 0
        self.hom = 10e-8

    def zero_grad(self):
        for layer in self.model.layers:
            for i in range(len(layer.cached_grad)):
                layer.grads.append({"s":0, "r":0})

    def step(self):

        self.num_step += 1

        for layer in self.model.layers:
            # init
            if len(layer.grads) != len(layer.cached_grad):
                self.zero_grad()

            for i in range(len(layer.grads)):
                grad = layer.cached_grad[i]
                s = layer.grads[i]["s"]
                r = layer.grads[i]["r"]

                s = self.fy1 * s + (1 - self.fy1) * grad
                r = self.fy2 * r + (1 - self.fy2) * grad * grad
                s_hat = s / (1-self.fy1**(self.num_step+1))
                r_hat = r / (1-self.fy2**(self.num_step+1))
                theta = - self.lr * (s_hat / (r_hat ** 0.5 + self.hom))
                layer.weights[i] += theta