from simpleNet.layers.Layer import Layer
import numpy as np

class MaxPooling2D(Layer):
    def __init__(self, stride):
        super().__init__()
        self.name = "MaxPooling2D"
        if isinstance(stride, int):
            self.stride_h = self.stride_w = stride
        else:
            self.stride_h, self.stride_w = stride

        super().__init__()

    def __call__(self, *args, **kwargs):
        x = args[0]
        m, c, in_h, in_w = x.shape
        assert in_h % self.stride_h == 0 and in_w % self.stride_w == 0
        self.cached_x = x
        out_h = int(in_h / self.stride_h)
        out_w = int(in_w / self.stride_w)
        self.mask = np.zeros(x.shape, dtype=np.float32)
        y = np.zeros((m, c, out_h, out_w), dtype=np.float32)

        for i in range(out_h):
            for j in range(out_w):
                top, bottom = i*self.stride_h, (i+1)*self.stride_h
                left, right = j*self.stride_w, (j+1)*self.stride_w
                temp = x[:, :, top:bottom, left:right]
                max_value = np.max(temp, (-1, -2), keepdims=True)
                self.mask[:, :, top:bottom, left:right] = (temp == max_value).astype(np.float32)
                y[:, :, i, j] = np.sum(temp * self.mask[:, :, top:bottom, left:right] , axis=(-1, -2), keepdims=False)

        return y

    def backwards(self, da):
        # 从da往x传播
        x = self.cached_x
        m, channels, in_h, in_w = x.shape
        assert da.shape[:2] == (m, channels)
        out_h, out_w = da.shape[2:4]
        dx = np.zeros(x.shape, dtype=np.float32)

        for i in range(out_h):
            for j in range(out_w):
                top, bottom = i * self.stride_h, (i + 1) * self.stride_h
                left, right = j * self.stride_w, (j + 1) * self.stride_w
                mask_area = self.mask[:, :, top:bottom, left:right]
                da_area = np.expand_dims(np.expand_dims(da[:, :, i, j], axis=-1), axis=-1)
                dx[:, :, top:bottom, left:right] =  mask_area * da_area
        return dx

    def __str__(self):
        return "%s: stride: (%d, %d)" % (self.name, self.stride_h, self.stride_w)
