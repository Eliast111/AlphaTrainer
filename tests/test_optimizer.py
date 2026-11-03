import unittest

from neural.neuron import Neuron
from neural.layer import Layer
from neural.network import Network
from learning.backprop import backpropagate
from learning.loss import mse
from learning.optimizer import GradientDescent


class TestAlphaOptimizer(unittest.TestCase):

    def create_network(self):
        return Network([
            Layer([
                Neuron([0.0], bias=0.0, activation="linear")
            ])
        ])

    def test_weight_update(self):
        network = self.create_network()
        optimizer = GradientDescent(0.1)

        optimizer.step(network, [[{
            "weights": [2.0],
            "bias": 0.0
        }]])

        self.assertAlmostEqual(
            network.layers[0].neurons[0].weights[0],
            -0.2
        )

    def test_bias_update(self):
        network = self.create_network()
        optimizer = GradientDescent(0.1)

        optimizer.step(network, [[{
            "weights": [0.0],
            "bias": 3.0
        }]])

        self.assertAlmostEqual(
            network.layers[0].neurons[0].bias,
            -0.3
        )

    def test_invalid_learning_rate(self):
        with self.assertRaises(ValueError):
            GradientDescent(0)

    def test_invalid_gradient_shape(self):
        network = self.create_network()
        optimizer = GradientDescent(0.1)

        with self.assertRaises(ValueError):
            optimizer.step(network, [])

    def test_learning_reduces_error(self):
        network = self.create_network()
        optimizer = GradientDescent(0.1)

        inputs = [1.0]
        targets = [2.0]

        initial_prediction = network.forward(inputs)
        initial_error = mse(initial_prediction, targets)

        result = backpropagate(network, inputs, targets)
        optimizer.step(network, result["gradients"])

        final_prediction = network.forward(inputs)
        final_error = mse(final_prediction, targets)

        self.assertLess(final_error, initial_error)


if __name__ == "__main__":
    unittest.main()
