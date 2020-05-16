from simpleNet.layers.Layer import Layer
import numpy as np


class Concate(Layer):
    def __init__(self, axis=-1):
        super().__init__()
        self.name = "Concate"
        self.axis = axis

    def __call__(self, mats):
        y = np.concatenate(mats, axis=self.axis)
        self.caches = mats
        return y

    def backwards(self, da):
        mats = self.caches
        axis = self.axis

        now = 0
        dmats = []
        for mat in mats:
            mat_inds = [slice(None, None, None)] * len(mat.shape)  # 建立全取引索
            mat_inds[axis] = slice(now, now + mat.shape[axis], None)  # 目标维度替换
            dmat = da[tuple(mat_inds)]
            dmats.append(dmat)
            now += mat.shape[axis]

        return tuple(dmats)

    def __str__(self):
        return self.name
