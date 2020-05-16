class Layer:
    def __init__(self):
        # 和正反传播相关的必备参数和缓存
        self.weights = {}
        self.cached_grad = {}
        self._mode = "train"
        self.name = ""

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

    @property
    def grads(self):
        return self.cached_grad

    def backwards(self, da):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()
