import numpy as np
from simpleNet import layers

a = np.arange(1 * 1 * 4 * 4, dtype=np.float32).reshape((1, 1, 4, 4))
pool = layers.MaxPooling2D(2)

print(a)
print(pool(a).shape)
print(pool(a)[0, 0, :, :])


