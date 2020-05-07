from simpleNet.layers.Layer import Layer
import numpy as np

"""
    持有各层
    初始化
    安排训练
"""


class Moduel(Layer):

    def __init__(self, layers=None):
        super().__init__()
        if layers:
            self.layers = layers
        else:
            self.layers = []
        self.name = "moduel"

    @property
    def weights(self):
        dic = {}
        for layer in self.layers:
            dic[layer.name] = layer.weights
        return dic

    @weights.setter
    def weights(self, value):
        for layer_name in value:
            for layer in self.layers:
                if layer.name == layer_name:
                    layer.weights = value[layer_name]

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value != self._mode:
            for layer in self.layers:
                layer.mode = value
            self._mode = value

    def __setattr__(self, key, value):
        # 添加layer
        if isinstance(value, Layer):
            self.layers.append(value)
            value.name = key
        object.__setattr__(self, key, value)

    def __call__(self, *args, **kwargs):
        if "mode" in kwargs:
            self.mode = kwargs['mode']
        return self.forwards(*args)

    def __str__(self):
        summary = self.name + ": { \n"
        for layer in self.layers:
            layer_summery = str(layer)
            lines = layer_summery.split("\n")
            for single_line in lines:
                summary += "  " + single_line + "\n"
        summary += "}"
        return summary

    def forwards(self, args):
        x = args
        for layer in self.layers:
            x = layer(x)
        return x

    def backwards(self, da):
        da_prev = da
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            da_prev = layer.backwards(da_prev)
        return da_prev

    def save_weights(self, path):
        weights = self.weights
        np.savez(path, weights=weights)

    def load_weights(self, path):
        db = np.load(path, allow_pickle=True)
        weights = db["weights"]
        dic = dict(np.ndenumerate(weights))
        real_weights = None
        for key in dic:
            real_weights = dic[key]

        self.weights = real_weights

    def summary(self):
        print(str(self))
