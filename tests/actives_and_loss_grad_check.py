import numpy as np
from simpleNet import layers, losses


conv = layers.Dense(2, 3, bias=False)
x = np.random.rand(5, 2) #.astype(np.float64)
y = np.random.rand(5, 3) #.astype(np.float64)
N = x.shape[0]
epsilon = 1e-4


def binary_cross_entropy(y_pred, y):
    logprobs = -np.log(y_pred) * y + -np.log(1 - y_pred) * (1 - y)
    cost = np.sum(logprobs) / y_pred.shape[0]
    return cost


def binary_cross_entropy_back(y_pred, y):
    return (1 - y) / (1 - y_pred) - (y / y_pred)


def category_cross_entropy(y_pred, y):
    epsilon = 1e-12
    predictions = np.clip(y_pred, epsilon, 1. - epsilon)
    ce = -np.sum(y * np.log(y_pred)) / y.shape[0]
    return ce

137
def category_cross_entropy_back(y_pred, y):
    return - y / y_pred


def mean_squared_error(y_pred, y):
    return 0.5 * np.sum(np.square(y_pred - y)) / y_pred.shape[0]


def mean_squared_error_back(y_pred, y):
    return y_pred - y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_back(a, y):
    return a * (1 - a) * y


def relu(z):
    return np.maximum(0, z)


def relu_back(a, da):
    dz = np.copy(da)
    dz[a<=0] = 0
    return dz


def my_loss(y_pred, y):
    return np.sum(y_pred) / y.shape[0]


def my_loss_back(y_pred, y):
    return np.ones(y_pred.shape)


# grad
z = conv(x)
a = relu(z)
l = my_loss(a, y)
da = my_loss_back(a, y)
dz = relu_back(a, da)

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
        a_plus = relu(z_plus)
        J_plus = my_loss(a_plus, y)  ######

        conv.weights["w"] = w_minus
        z_minus = conv(x)
        a_minus = relu(z_minus)
        J_minus = my_loss(a_minus, y)  #####
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