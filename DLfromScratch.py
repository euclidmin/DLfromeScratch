import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist


y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

# MSE
# y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
mse = mean_squared_error(np.array(y1), np.array(t))
print(mse)

# y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
mse = mean_squared_error(np.array(y2), np.array(t))
print(mse)

#CEE
cee = cross_entropy_error(np.array(y1), np.array(t))
print(cee)

cee = cross_entropy_error(np.array(y2), np.array(t))
print(cee)
