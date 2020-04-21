import numpy as np
from simpleNet.utils.im2col_tool import im2col, col2im

a = np.arange(2 * 2 * 3 * 3)
a = np.reshape(a, (2, 2, 3, 3))
w = np.arange(3 * 2 * 2 * 2)
w = np.reshape(w, (3, 2, 2, 2))
print(a)

b, w2 = im2col(a, w, 1, 1)
print(b)

print("-----------------------")

print(w)
print(w2)
