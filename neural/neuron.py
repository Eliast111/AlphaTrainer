"""
Alpha Neural Engine V0.2
Neuronio artificial com suporte a gradientes.
"""

from alpha_math.activation import (
    linear, relu, sigmoid, tanh, leaky_relu,
    linear_derivative, relu_derivative,
    sigmoid_derivative, tanh_derivative, leaky_relu_derivative
)

ACTIVATIONS = {
    "linear": linear,
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "leaky_relu": leaky_relu,
}

DERIVATIVES = {
    "linear": linear_derivative,
    "relu": relu_derivative,
    "sigmoid": sigmoid_derivative,
    "tanh": tanh_derivative,
    "leaky_relu": leaky_relu_derivative,
}


class Neuron:
    def __init__(self, weights, bias=0.0, activation="relu"):
        if not weights:
            raise ValueError("O neuronio precisa de pelo menos um peso.")

        if activation not in ACTIVATIONS:
            raise ValueError(
                "Funcao de ativacao desconhecida: "
                f"{activation!r}. Disponiveis: {', '.join(sorted(ACTIVATIONS))}."
            )

        self.weights = [float(w) for w in weights]
        self.bias = float(bias)
        self.activation_name = activation
        self.activation = ACTIVATIONS[activation]
        self.activation_derivative = DERIVATIVES[activation]

    def forward_with_cache(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError("Numero de entradas incompativel com os pesos.")

        total = sum(
            float(value) * weight
            for value, weight in zip(inputs, self.weights)
        )

        z = total + self.bias
        output = self.activation(z)

        return output, z

    def forward(self, inputs):
        output, _ = self.forward_with_cache(inputs)
        return output
