"""
Alpha Activation Engine V0.1
Funcoes de ativacao para redes neurais.
"""

import math


def relu(x):
    return max(0.0, x)


def relu_derivative(x):
    return 1.0 if x > 0 else 0.0


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))

    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def sigmoid_derivative(x):
    value = sigmoid(x)
    return value * (1.0 - value)


def tanh(x):
    return math.tanh(x)


def tanh_derivative(x):
    value = tanh(x)
    return 1.0 - value * value


def linear(x):
    return float(x)


def linear_derivative(x):
    return 1.0


LEAKY_SLOPE = 0.01


def leaky_relu(x):
    return x if x > 0 else LEAKY_SLOPE * x


def leaky_relu_derivative(x):
    return 1.0 if x > 0 else LEAKY_SLOPE
