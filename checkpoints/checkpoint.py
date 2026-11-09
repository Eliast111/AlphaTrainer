"""
Alpha Checkpoint Engine V0.1
Guardar e carregar modelos neuronais em JSON.
"""

import json
import math
import os
import tempfile

from neural.neuron import Neuron
from neural.layer import Layer
from neural.network import Network


def write_json_atomic(data, path):
    """Escreve JSON de forma atomica (nao deixa ficheiros a meio)."""
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    handle, temp_path = tempfile.mkstemp(
        dir=directory, prefix=".alpha_", suffix=".tmp"
    )

    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

        os.replace(temp_path, path)
    except BaseException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def save_model(model, path, metadata=None):
    if not isinstance(model, Network):
        raise TypeError("O objeto deve ser uma Network")

    data = {
        "format": "alpha_checkpoint_v1",
        "layers": []
    }

    for layer in model.layers:
        layer_data = []

        for neuron in layer.neurons:
            values = neuron.weights + [neuron.bias]

            if not all(math.isfinite(value) for value in values):
                raise ValueError("O modelo contém valores não finitos")

            layer_data.append({
                "weights": neuron.weights,
                "bias": neuron.bias,
                "activation": neuron.activation_name
            })

        data["layers"].append(layer_data)

    if metadata is not None:
        if not isinstance(metadata, dict):
            raise TypeError("metadata deve ser um dicionario.")

        data["metadata"] = metadata

    write_json_atomic(data, path)


def load_model(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Checkpoint inválido")

    if data.get("format") != "alpha_checkpoint_v1":
        raise ValueError("Formato de checkpoint desconhecido")

    saved_layers = data.get("layers")

    if not isinstance(saved_layers, list) or not saved_layers:
        raise ValueError("Estrutura de camadas inválida")

    layers = []

    for saved_layer in saved_layers:
        if not isinstance(saved_layer, list) or not saved_layer:
            raise ValueError("Camada inválida")

        neurons = []

        for item in saved_layer:
            if not isinstance(item, dict):
                raise ValueError("Dados do neurónio inválidos")

            weights = item.get("weights")
            bias = item.get("bias")
            activation = item.get("activation")

            if not isinstance(weights, list) or not weights:
                raise ValueError("Pesos inválidos")

            if not isinstance(bias, (int, float)) or isinstance(bias, bool):
                raise ValueError("Bias inválido")

            if not all(
                isinstance(w, (int, float))
                and not isinstance(w, bool)
                and math.isfinite(w)
                for w in weights
            ):
                raise ValueError("Peso inválido")

            if not math.isfinite(bias):
                raise ValueError("Bias não finito")

            neurons.append(
                Neuron(
                    weights=weights,
                    bias=bias,
                    activation=activation
                )
            )

        layers.append(Layer(neurons))

    for previous, current in zip(layers, layers[1:]):
        expected = len(previous.neurons)
        actual = len(current.neurons[0].weights)

        if any(len(n.weights) != expected for n in current.neurons):
            raise ValueError("Dimensões das camadas incompatíveis")

    return Network(layers)


def read_metadata(path):
    """Metadados guardados com o modelo ({} se nao existirem)."""
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Checkpoint invalido")

    metadata = data.get("metadata", {})

    if not isinstance(metadata, dict):
        raise ValueError("Metadados invalidos")

    return metadata
