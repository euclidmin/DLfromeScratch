import numpy as np
import sys, os
sys.path.append("D:\\dev\\projects\\deep-learning-from-scratch")
sys.path.append("D:\\dev\\projects\\deep-learning-from-scratch\\ch04")
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

print(net.params['W1'].shape)
print(net.params['b1'].shape)
print(net.params['W2'].shape)
print(net.params['b2'].shape)

x = np.random.rand(100, 784)
y = net.predict(x)
t = np.random.rand(100, 10)

grads = net.numerical_gradient(x, t)

print(grads['W1'].shape)
print(grads['b1'].shape)
print(grads['W2'].shape)
print(grads['b2'].shape)