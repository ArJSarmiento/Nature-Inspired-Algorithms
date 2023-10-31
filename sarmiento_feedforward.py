"""
Programmer Information:
  Name: Arnel Jan E. Sarmiento
  Course: 3BSCS
  Student No.: 2021-05094

Program Description:
  Implement the feedforward process for 2-input, 1-hidden layer (with 2 neurons in the hidden layer)
  neural network given one set of input.
"""

import numpy as np


def sigmoid(x):
    # Our activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # Weight inputs, add bias, then use the activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


if __name__ == "__main__":
    # h_1----------------------------------------------------------------
    weights = np.array([0.5, 0.5])  # w1 = 0.5, w2 = 0.5
    bias = 2  # b = 2
    h_1 = Neuron(weights, bias)

    # h_1----------------------------------------------------------------
    weights = np.array([0.75, 0.75])  # w1 = 0.75, w2 = 0.75
    bias = 3  # b = 3
    h_2 = Neuron(weights, bias)

    # o_1------------------------------------------------------------------
    weights = np.array([0, 0.5])  # w1 = 0, w2 = 0.5
    bias = 4  # b = 4
    o_1 = Neuron(weights, bias)

    x = np.array([2, 3])  # x1 = 2, x2 = 3

    print("h_1 output: ", h_1.feedforward(x))
    print("h_2 output: ", h_2.feedforward(x))
    print("o_1 output: ", o_1.feedforward(np.array([h_1.feedforward(x), h_2.feedforward(x)])))
