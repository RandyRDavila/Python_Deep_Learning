
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

from mlp import MultilayerPerceptron



if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X/255
    test_X = test_X/255

    # X will temp store flattened matrices
    X = []
    for x in train_X:
        X.append(x.flatten().reshape(784, 1))

    # Y will temp store one-hot encoded label vectors
    Y = []
    for y in train_y:
        temp_vec = np.zeros((10, 1))
        temp_vec[y][0] = 1.0
        Y.append(temp_vec)

    # Our data will be stored as a list of tuples. 
    train_data = [p for p in zip(X, Y)]

    # The same above for testing data 
    X = []
    for x in test_X:
        X.append(x.flatten().reshape(784, 1))

    Y = []
    for y in test_y:
        temp_vec = np.zeros((10, 1))
        temp_vec[y][0] = 1.0
        Y.append(temp_vec)

    test_data = [p for p in zip(X, Y)]


    net = MultilayerPerceptron()

    print(f"Before training error = {net.MSE(test_data)}")
    net.stochastic_gradient_descent(train_data)
    print("")
    print(f"After training error = {net.MSE(test_data)}")