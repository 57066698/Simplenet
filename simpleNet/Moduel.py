from simpleNet.layers.Layer import Layer

"""
    持有各层
    初始化
    安排训练
"""


class Moduel(Layer):

    def __init__(self):
        super().__init__()
        self.layers = []

    def __setattr__(self, key, value):
        if isinstance(value, Layer):
            self.layers.append(value)
            value.name = key
        self.__dict__[key] = value

    def __call__(self, *args, **kwargs):
        if "run" in kwargs and kwargs["run"] == True:
            self.change_state("run")
        else:
            self.change_state("train")
        return self.forwards(*args)

    def __str__(self):
        summary = "moduel: { \n"
        for layer in self.layers:
            layer_summery = str(layer)
            lines = layer_summery.split("\n")
            for single_line in lines:
                summary += "  " + single_line + "\n"
        summary += "}"
        return summary

    def change_state(self, statu):
        if self.statu == statu:
            return
        assert statu in ["train", "run"]

        for layer in self.layers:
            layer.change_state(statu)
        self.statu = statu

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

    def summary(self):
        print(str(self))
