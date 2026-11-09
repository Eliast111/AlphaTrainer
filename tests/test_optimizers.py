import unittest

from learning.optimizer import (
    GradientDescent, Momentum, Adam, create_optimizer
)
from neural.neuron import Neuron
from neural.layer import Layer
from neural.network import Network
from training.trainer import Trainer


def line_network():
    return Network([Layer([Neuron([0.0], bias=0.0, activation="linear")])])


DATA = [([x / 4.0], [2.0 * (x / 4.0) + 1.0]) for x in range(5)]


class TestOptimizers(unittest.TestCase):

    def converge(self, optimizer, epochs=300):
        network = line_network()
        result = Trainer(optimizer).fit(network, DATA, epochs=epochs)
        return result["final_loss"]

    def test_gradient_descent(self):
        self.assertLess(self.converge(GradientDescent(0.2), 400), 1e-3)

    def test_momentum(self):
        self.assertLess(self.converge(Momentum(0.05, 0.9), 300), 1e-3)

    def test_adam(self):
        self.assertLess(self.converge(Adam(0.1), 300), 1e-3)

    def test_create_optimizer(self):
        self.assertIsInstance(create_optimizer("adam"), Adam)
        self.assertIsInstance(create_optimizer("SGD", 0.5), GradientDescent)
        self.assertEqual(create_optimizer("momentum", 0.3).learning_rate, 0.3)

    def test_unknown_optimizer(self):
        with self.assertRaises(ValueError):
            create_optimizer("rmsprop")

    def test_invalid_hyperparameters(self):
        with self.assertRaises(ValueError):
            Adam(0)
        with self.assertRaises(ValueError):
            Adam(0.1, beta1=1.0)
        with self.assertRaises(ValueError):
            Momentum(0.1, momentum=-0.1)

    def test_invalid_gradients_do_not_change_parameters(self):
        network = line_network()
        bad = [[{"weights": [float("nan")], "bias": 0.0}]]
        with self.assertRaises(ValueError):
            Adam(0.1).step(network, bad)
        self.assertEqual(network.layers[0].neurons[0].weights, [0.0])


if __name__ == "__main__":
    unittest.main()
