"""
Alpha Model Factory V0.1
Criacao automatica de redes neuronais.
"""

from neural.neuron import Neuron
from neural.layer import Layer
from neural.network import Network


def create_model(layer_sizes, activation="relu"):
    if not isinstance(layer_sizes, (list, tuple)):
        raise ValueError("layer_sizes deve ser uma lista ou tupla")

    if len(layer_sizes) < 2:
        raise ValueError("O modelo precisa de pelo menos duas camadas")

    if any(
        not isinstance(size, int) or isinstance(size, bool) or size <= 0
        for size in layer_sizes
    ):
        raise ValueError("Todos os tamanhos devem ser inteiros positivos")

    layers = []

    for index in range(1, len(layer_sizes)):
        input_size = layer_sizes[index - 1]
        neuron_count = layer_sizes[index]

        neurons = [
            Neuron(
                [0.01] * input_size,
                bias=0.0,
                activation=activation
            )
            for _ in range(neuron_count)
        ]

        layers.append(Layer(neurons))

    return Network(layers)
