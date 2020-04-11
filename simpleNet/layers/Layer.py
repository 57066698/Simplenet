class Layer:
    def __init__(self):
        # 和正方传播相关的必备参数和缓存
        self.weights = []
        self.grads = []
        self.cached_grad = None
        self.status = "train"
        pass

    def backwards(self, da):
        pass
