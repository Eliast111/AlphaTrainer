import json
import unittest

from neural.neuron import Neuron
from neural.layer import Layer
from neural.network import Network
from learning.optimizer import (
    GradientDescent,
    Momentum,
    Adam,
)


def make_network():
    return Network([
        Layer([
            Neuron([0.5, -0.3], bias=0.1, activation="linear"),
            Neuron([-0.2, 0.8], bias=-0.1, activation="linear"),
        ]),
        Layer([
            Neuron([0.4, -0.6], bias=0.2, activation="linear"),
        ]),
    ])


def make_gradients():
    return [
        [
            {"weights": [0.2, -0.4], "bias": 0.1},
            {"weights": [-0.3, 0.5], "bias": -0.2},
        ],
        [
            {"weights": [0.6, -0.1], "bias": 0.3},
        ],
    ]


class TestOptimizerState(unittest.TestCase):

    def configurations(self):
        return [
            (
                GradientDescent(0.05),
                GradientDescent(),
            ),
            (
                Momentum(0.05, momentum=0.8),
                Momentum(),
            ),
            (
                Adam(0.01, beta1=0.85, beta2=0.99),
                Adam(),
            ),
        ]

    def assert_networks_equal(self, first, second):
        for layer_a, layer_b in zip(first.layers, second.layers):
            for neuron_a, neuron_b in zip(
                layer_a.neurons, layer_b.neurons
            ):
                self.assertEqual(neuron_a.weights, neuron_b.weights)
                self.assertEqual(neuron_a.bias, neuron_b.bias)

    def test_json_round_trip_and_resume(self):
        for original_optimizer, restored_optimizer in self.configurations():
            with self.subTest(optimizer=type(original_optimizer).__name__):
                original_network = make_network()

                # Criar histórico no otimizador.
                original_optimizer.step(
                    original_network, make_gradients()
                )
                original_optimizer.step(
                    original_network, make_gradients()
                )

                # Simular gravação e leitura JSON.
                saved_state = original_optimizer.state_dict(
                    original_network
                )
                encoded = json.dumps(saved_state)
                decoded = json.loads(encoded)

                # Reconstruir a rede com os parâmetros guardados.
                restored_network = make_network()

                for source_layer, target_layer in zip(
                    original_network.layers,
                    restored_network.layers
                ):
                    for source, target in zip(
                        source_layer.neurons,
                        target_layer.neurons
                    ):
                        target.weights = list(source.weights)
                        target.bias = source.bias

                restored_optimizer.load_state_dict(
                    decoded, restored_network
                )

                # Continuar ambos com os mesmos gradientes.
                gradients = make_gradients()

                original_optimizer.step(
                    original_network, gradients
                )
                restored_optimizer.step(
                    restored_network, gradients
                )

                self.assert_networks_equal(
                    original_network, restored_network
                )

                self.assertEqual(
                    original_optimizer.state_dict(original_network),
                    restored_optimizer.state_dict(restored_network),
                )

    def test_rejects_incompatible_network(self):
        optimizer = Momentum(0.05)
        network = make_network()
        state = optimizer.state_dict(network)

        incompatible = Network([
            Layer([
                Neuron([0.1], activation="linear")
            ])
        ])

        with self.assertRaises(ValueError):
            optimizer.load_state_dict(state, incompatible)

    def test_rejects_wrong_optimizer(self):
        network = make_network()
        state = Adam().state_dict(network)

        with self.assertRaises(ValueError):
            Momentum().load_state_dict(state, network)

    def test_invalid_adam_state_does_not_change_optimizer(self):
        network = make_network()
        optimizer = Adam(0.02)
        state = optimizer.state_dict(network)

        state["beta1"] = 2.0

        with self.assertRaises(ValueError):
            optimizer.load_state_dict(state, network)

        self.assertEqual(optimizer.learning_rate, 0.02)
        self.assertEqual(optimizer.beta1, 0.9)


if __name__ == "__main__":
    unittest.main()
