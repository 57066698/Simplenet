from simpleNet.layers.Layer import Layer

"""
    持有各层
    初始化
    安排训练
"""


class Moduel:

    def __init__(self):
        self.layers = []
        self.states = "train"

    def __setattr__(self, key, value):
        if isinstance(value, Layer):
            self.layers.append(value)
        self.__dict__[key] = value

    def __call__(self, *args, **kwargs):
        inputs = args

        if "run" in kwargs and kwargs["run"]:
            self.change_state("run")
        else:
            self.change_state("train")
        self.forwards(*args)

    def forwards(self, args):
        pass

    def backwards(self, da):
        da_prev = da
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            da_prev = layer.backwards(da_prev)

    def change_state(self, state):
        if self.states == state:
            return
        assert state in ["train", "run"]
        for layer in self.layers:
            layer.status = state

