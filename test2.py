import torch
import numpy as np

a = np.array([[2, 1, 0], [1, 0, 3]])
b = np.zeros((a.shape[0], a.shape[1], a.max()+1))
shape = b.shape
b = b.reshape((-1, shape[2]))
a = a.reshape(-1)

b[np.arange(b.shape[0]), a] = 1
b = b.reshape(shape)


print(b)