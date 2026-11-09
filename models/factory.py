"""
Alpha Model Factory V0.2
Criacao automatica de redes neuronais com inicializacao aleatoria
(Xavier / He), reproduzivel atraves de uma seed.
"""

import math
import random

from neural.neuron import Neuron, ACTIVATIONS
from neural.layer import Layer
from neural.network import Network

INITIALIZERS = ("auto", "xavier", "he", "constant")


def _limit(init, activation, fan_in, fan_out):
    if init == "auto":
        init = "he" if activation in ("relu", "leaky_relu") else "xavier"

    if init == "he":
        return math.sqrt(6.0 / fan_in)

    return math.sqrt(6.0 / (fan_in + fan_out))


def create_model(
    layer_sizes,
    activation="relu",
    output_activation=None,
    init="auto",
    seed=42
):
    """Cria uma rede totalmente ligada.

    layer_sizes: [entradas, oculta1, ..., saida]
    activation: ativacao das camadas ocultas.
    output_activation: ativacao da saida (por omissao igual a activation).
    init: auto | xavier | he | constant (0.01, comportamento antigo).
    seed: torna a inicializacao reproduzivel (None = aleatoria).
    """
    if not isinstance(layer_sizes, (list, tuple)):
        raise ValueError("layer_sizes deve ser uma lista ou tupla")

    if len(layer_sizes) < 2:
        raise ValueError("O modelo precisa de pelo menos duas camadas")

    if any(
        not isinstance(size, int) or isinstance(size, bool) or size <= 0
        for size in layer_sizes
    ):
        raise ValueError("Todos os tamanhos devem ser inteiros positivos")

    if init not in INITIALIZERS:
        raise ValueError(
            "Inicializacao desconhecida. Disponiveis: "
            + ", ".join(INITIALIZERS)
        )

    if output_activation is None:
        output_activation = activation

    for name in (activation, output_activation):
        if name not in ACTIVATIONS:
            raise ValueError(f"Funcao de ativacao desconhecida: {name!r}")

    generator = random.Random(seed)
    layers = []
    total = len(layer_sizes) - 1

    for index in range(1, len(layer_sizes)):
        input_size = layer_sizes[index - 1]
        neuron_count = layer_sizes[index]
        layer_activation = (
            output_activation if index == total else activation
        )

        limit = _limit(init, layer_activation, input_size, neuron_count)

        neurons = []

        for _ in range(neuron_count):
            if init == "constant":
                weights = [0.01] * input_size
            else:
                weights = [
                    generator.uniform(-limit, limit)
                    for _ in range(input_size)
                ]

            neurons.append(
                Neuron(weights, bias=0.0, activation=layer_activation)
            )

        layers.append(Layer(neurons))

    return Network(layers)
