from simpleNet.core.Layer import Layer

"""
    持有各层
    初始化
    安排训练
"""


class Moduel:

    def __init__(self):
        self.layers = []

    def __setattr__(self, key, value):
        if isinstance(value, Layer):
            self.layers.append(value)
        self.__dict__[key] = value

    def forwards(self):
        pass

    def backwards(self, da):

        da_prev = da
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            da_prev = layer.backwards(da_prev)

    def learn(self, lr:float):

        for layer in self.layers:
            layer.learn(lr)
