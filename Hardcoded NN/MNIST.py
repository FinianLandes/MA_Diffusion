from NN import *
from keras.datasets import mnist
import logging


n_test_points: int = 400
(train, label_train), (test, label_test) = mnist.load_data()
train = (np.array(train).reshape(len(train), 1, 28, 28) / 255.0)[:n_test_points]
label_train = one_hot_encode(label_train[:n_test_points], 10)
test = (np.array(test).reshape(len(test), 1, 28, 28) / 255.0)[:100]
label_test = one_hot_encode(label_test, 10)[:100]
architecture: list[Layer] = [Convolution((1, 28, 28), 3, 16), Max_Pool(3), Up_Convolution((16, 9, 9), 3, 6), Reshape((16, 9, 9),  (16*9*9, 1)), Dense_Layer(16*9*9, 10), Sigmoid()]
#architecture: list[Layer] = [Dense_Layer(28*28, 40), Sigmoid(), Dense_Layer(40, 40), Sigmoid(), Dense_Layer(40, 16), Sigmoid(), Dense_Layer(16, 10), Sigmoid()]
m = Model("MNIST_v2", architecture, MSE(), 0.05, logging.DEBUG, True)
m.train(train, label_train, 20)
m.classification_test(test, label_test)

