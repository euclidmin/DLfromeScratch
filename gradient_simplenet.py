import sys, os
# sys.path.append(os.pardir)
sys.path.append("D:\\dev\\projects\\deep-learning-from-scratch")
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet :
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(x)
        loss = cross_entropy_error(y, t)
        return loss

net = simpleNet()
print(net.W)

x12 = np.array([0.6, 0.9])
p13 = net.predict(x12) # 12 X 23 = 13
print(p13)

max = np.argmax(p13)
print(max) # index 출력

t13 = np.array([0, 0, 1])
loss = net.loss(x12, t13)
print(loss)


def f(W) :
    return net.loss(x12, t13)

dW = numerical_gradient(f, net.W)
print(dW)


