import numpy as np
from simpleNet import layers

# x 和 net
conv = layers.Conv2D(1, 1, 2, padding='valid', use_bias=False)
x = np.random.rand(1, 1, 2, 2)
epsilon = 1e-7
# grad
J = conv(x)
conv.backwards(J)
grad = conv.weights["w"]
# weight + - ，再算 J- J+
w_plus = np.copy(conv.weights["w"]) + epsilon
w_minus = np.copy(conv.weights["w"]) - epsilon
conv.weights["w"] = w_plus
J_plus = conv(x)
conv.weights["w"] = w_minus
J_minus = conv(x)
# gradapprox
gradapprox = (J_plus - J_minus) / (2*epsilon)
# 结果
numerator = np.linalg.norm(grad - gradapprox)
denominator = np.linalg.norm(gradapprox)
diff = numerator / denominator

print(diff)