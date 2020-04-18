class Layer:
    def __init__(self):
        # 和正反传播相关的必备参数和缓存
        self.weights = []
        self.cached_grad = []
        self.statu = "train"
        self.name = ""

    def backwards(self, da):
        raise NotImplementedError()

    def change_state(self, statu):
        if self.statu == statu:
            return
        assert statu in ["train", "run"]
        self.statu = statu

    def __repr__(self):
        print(self.__str__())

    def __str__(self):
        raise NotImplementedError()
