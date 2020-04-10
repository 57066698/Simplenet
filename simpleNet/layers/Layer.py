class Layer:
    def __init__(self):
        # 和正方传播相关的必备参数和缓存
        self.weights = []
        self.grads = []
        self.cached_x = None
        self.cached_grad = None
        pass

    def backwards(self):
        pass

    def learn(self, lr: float):
        pass
