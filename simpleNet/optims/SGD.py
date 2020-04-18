"""
    adam optimizer
    * 持有层
    * da 记录在各层
    * momentum, velocity 也记录在各层
"""
from simpleNet.layers import Layer
from simpleNet import Moduel

class SGD:
    def __init__(self, model:Layer, lr:float=0.001):
        self.model = model
        self.lr = lr

    def step(self):

        def step_layer(layer):
            for i in range(len(layer.cached_grad)):
                grad = layer.cached_grad[i]
                layer.weights[i] -= self.lr * grad

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

