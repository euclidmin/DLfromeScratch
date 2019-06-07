import numpy as np
import sys, os
from pathlib import Path
# sys.path.append('C:\\dev\\PycharmProjects\\DLfromScratch\\deep-learnig-from-scratch')
# print(sys.path)
from dataset.mnist import load_mnist
from PIL import Image
import pickle
from functions import sigmoid


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = sigmoid(a3)

    return y

def main():
    x, t = get_data()
    print(x.shape)
    print(t.shape)
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1

    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


if __name__ == '__main__':
    main()



#
#
# img = x_train[0]
# label = t_train[0]
# print(label)
#
# print(img.shape)
# img = img.reshape(28, 28)
# print(img.shape)
#
# img_show(img)
#


# y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
#
# y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
#
#
# def mean_squared_error(y, t):
#     return 0.5 * np.sum((y - t) ** 2)
#
#
# def cross_entropy_error(y, t):
#     # print(y.ndim)
#     # print(y)
#     if y.ndim == 1 :
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#
#     # print(y.ndim)
#     # print(y)
#     batch_size = y.shape[0]
#     delta = 1e-7
#     return -np.sum(t * np.log(y + delta))/batch_size
#
# # MSE
# # y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# mse = mean_squared_error(np.array(y1), np.array(t))
# print(mse)
# # y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# mse = mean_squared_error(np.array(y2), np.array(t))
# print(mse)
#
# #CEE
# cee = cross_entropy_error(np.array(y1), np.array(t))
# print(cee)
# cee = cross_entropy_error(np.array(y2), np.array(t))
# print(cee)
#
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# print(x_train.shape) #(60000, 784)
# print(t_train.shape) #(60000, 10)
#
# train_size = x_train.shape[0] # 60000
# batch_size = t_train.shape[0] # 10
# batch_mask = np.random.choice(train_size, batch_size)
# print(batch_mask)
# x_batch = x_train[batch_mask]
# t_batch = t_train[batch_mask]
#


