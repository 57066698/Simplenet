import numpy as np
from simpleNet import layers, losses

# x 和 net
conv = layers.Dense(2, 3, use_bias=False)
loss = losses.MeanSquaredError()
x = np.random.rand(5, 2) #.astype(np.float64)
y = np.random.rand(5, 3) #.astype(np.float64)
N = x.shape[0]
epsilon = 1e-4


def loss(y_pred, y):
    logprobs = -np.log(y_pred) * y + -np.log(1 - y_pred) * (1 - y)
    cost = np.sum(logprobs) / y_pred.shape[0]
    return cost


def loss_back(y_pred, y):
    return (1 - y) / (1 - y_pred) - (y / y_pred)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_back(a, y):
    return a * (1 - a) * y

# grad
z = conv(x)
a = sigmoid(z)
l = loss(a, y)
da = loss_back(a, y)
dz = sigmoid_back(a, da)

conv.backwards(dz)
grad = conv.cached_grad["w"]

# 计算gradapprox
w = np.copy(conv.weights["w"])
gradapprox = np.zeros(w.shape, dtype=np.float32)

for i in range(w.shape[0]):
    for j in range(w.shape[1]):
        # weight + - ，再算 J- J+
        w_plus = np.copy(w)
        w_plus[i][j] += epsilon
        w_minus = np.copy(w)
        w_minus[i][j] -= epsilon

        conv.weights["w"] = w_plus
        z_plus = conv(x)
        a_plus = sigmoid(z_plus)
        J_plus = loss(a_plus, y)

        conv.weights["w"] = w_minus
        z_minus = conv(x)
        a_minus = sigmoid(z_minus)
        J_minus = loss(a_minus, y)
        # gradapprox
        gradapprox[i][j] = (J_plus - J_minus) / (2 * epsilon)

# gradapprox /= 12
# 结果
numerator = np.linalg.norm(grad - gradapprox)
denominator = np.linalg.norm(gradapprox) + np.linalg.norm(grad)
diff = numerator / denominator

# print(grad)
# print(gradapprox)
print(diff)