from simpleNet import layers
import numpy as np

flatten = layers.Flatten()
x = np.arange(2*2*16)
x = np.reshape(x, (2, 2, 4, 4))

print(x.shape)
print(x[0, 1, 3, :])

y = flatten(x)
da = np.copy(y)

da_prev = flatten.backwards(da)
print(da_prev[0, 1, 3, :])
