import unittest

from neural.neuron import Neuron
from neural.layer import Layer
from neural.network import Network
from learning.backprop import backpropagate


class TestAlphaBackprop(unittest.TestCase):

    def test_single_neuron_gradient(self):
        network = Network([
            Layer([
                Neuron([2.0], bias=0.0, activation="linear")
            ])
        ])

        result = backpropagate(network, [3.0], [5.0])

        gradient = result["gradients"][0][0]

        self.assertAlmostEqual(gradient["weights"][0], 6.0)
        self.assertAlmostEqual(gradient["bias"], 2.0)

    def test_prediction_is_returned(self):
        network = Network([
            Layer([
                Neuron([2.0], bias=1.0, activation="linear")
            ])
        ])

        result = backpropagate(network, [3.0], [5.0])

        self.assertEqual(result["predictions"], [7.0])

    def test_two_layer_network(self):
        network = Network([
            Layer([
                Neuron([0.5, 0.5], activation="linear")
            ]),
            Layer([
                Neuron([1.0], activation="linear")
            ])
        ])

        result = backpropagate(network, [2.0, 4.0], [3.0])

        self.assertEqual(len(result["gradients"]), 2)
        self.assertEqual(len(result["gradients"][0]), 1)
        self.assertEqual(len(result["gradients"][1]), 1)

    def test_invalid_target_size(self):
        network = Network([
            Layer([
                Neuron([1.0], activation="linear")
            ])
        ])

        with self.assertRaises(ValueError):
            backpropagate(network, [2.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()


    def test_backprop_matches_numerical_gradient(self):
        from learning.gradients import numerical_gradient
        from learning.loss import mse

        parameters = [0.5, 0.5, 0.1, 1.2, -0.2]
        inputs = [2.0, 4.0]
        targets = [3.0]

        def build_network(values):
            return Network([
                Layer([
                    Neuron(
                        [values[0], values[1]],
                        bias=values[2],
                        activation="linear"
                    )
                ]),
                Layer([
                    Neuron(
                        [values[3]],
                        bias=values[4],
                        activation="linear"
                    )
                ])
            ])

        network = build_network(parameters)
        result = backpropagate(network, inputs, targets)

        analytical = [
            result["gradients"][0][0]["weights"][0],
            result["gradients"][0][0]["weights"][1],
            result["gradients"][0][0]["bias"],
            result["gradients"][1][0]["weights"][0],
            result["gradients"][1][0]["bias"]
        ]

        def calculate_loss(values):
            prediction = build_network(values).forward(inputs)
            return mse(prediction, targets)

        numerical = numerical_gradient(
            calculate_loss,
            parameters
        )

        for actual, expected in zip(analytical, numerical):
            self.assertAlmostEqual(actual, expected, places=4)
