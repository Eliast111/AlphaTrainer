"""
Alpha Learning Engine V0.3
Otimizador Gradient Descent.
"""

import math


class GradientDescent:
    def __init__(self, learning_rate=0.01):
        learning_rate = float(learning_rate)

        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate deve ser positivo e finito.")

        self.learning_rate = learning_rate

    def step(self, network, gradients):
        if len(gradients) != len(network.layers):
            raise ValueError("Numero de camadas incompativel.")

        # Validar todos os gradientes antes de alterar os parametros.
        updates = []

        for layer, layer_gradients in zip(network.layers, gradients):
            if len(layer_gradients) != len(layer.neurons):
                raise ValueError("Numero de neuronios incompativel.")

            for neuron, gradient in zip(layer.neurons, layer_gradients):
                if not isinstance(gradient, dict):
                    raise TypeError("Gradiente invalido.")

                weight_gradients = gradient.get("weights")
                bias_gradient = gradient.get("bias")

                if weight_gradients is None or bias_gradient is None:
                    raise ValueError("Gradientes incompletos.")

                if len(weight_gradients) != len(neuron.weights):
                    raise ValueError("Numero de pesos incompativel.")

                values = [float(value) for value in weight_gradients]
                values.append(float(bias_gradient))

                if not all(math.isfinite(value) for value in values):
                    raise ValueError("Os gradientes devem ser finitos.")

                updates.append((neuron, values))

        # Aplicar as atualizacoes.
        for neuron, values in updates:
            for index, gradient in enumerate(values[:-1]):
                neuron.weights[index] -= (
                    self.learning_rate * gradient
                )

            neuron.bias -= self.learning_rate * values[-1]
