import numpy as np
# import simpleNet

a_prev = np.array([[1, 2],[3, 4]])
w = np.array([[1], [2]])
y = [[10], [22]]

for i in range(1000):
    y_pred = np.matmul(a_prev, w)
    l = y_pred - y
    a_prev = a_prev - 0.01 * np.matmul(l, w.transpose())

print(np.matmul(a_prev, w))