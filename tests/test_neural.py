import unittest

from neural.neuron import Neuron


class TestAlphaNeuron(unittest.TestCase):

    def test_relu_forward(self):
        neuron = Neuron([0.5, 0.5], bias=0, activation="relu")
        self.assertEqual(neuron.forward([2, 4]), 3.0)

    def test_linear_forward(self):
        neuron = Neuron([2, 3], bias=1, activation="linear")
        self.assertEqual(neuron.forward([2, 3]), 14.0)

    def test_sigmoid_forward(self):
        neuron = Neuron([1], bias=0, activation="sigmoid")
        self.assertAlmostEqual(neuron.forward([0]), 0.5)

    def test_tanh_forward(self):
        neuron = Neuron([1], bias=0, activation="tanh")
        self.assertAlmostEqual(neuron.forward([0]), 0.0)

    def test_invalid_input_size(self):
        neuron = Neuron([1, 2])
        with self.assertRaises(ValueError):
            neuron.forward([5])

    def test_invalid_activation(self):
        with self.assertRaises(ValueError):
            Neuron([1], activation="unknown")


if __name__ == "__main__":
    unittest.main()


class TestAlphaLayer(unittest.TestCase):

    def test_layer_forward(self):
        from neural.layer import Layer

        n1 = Neuron([1, 1], bias=0, activation="linear")
        n2 = Neuron([2, 2], bias=0, activation="linear")

        layer = Layer([n1, n2])

        self.assertEqual(layer.forward([2, 3]), [5.0, 10.0])

    def test_layer_size(self):
        from neural.layer import Layer

        layer = Layer([
            Neuron([1]),
            Neuron([2]),
            Neuron([3])
        ])

        self.assertEqual(layer.size(), 3)

    def test_empty_layer(self):
        from neural.layer import Layer

        with self.assertRaises(ValueError):
            Layer([])

    def test_invalid_neuron(self):
        from neural.layer import Layer

        with self.assertRaises(TypeError):
            Layer(["invalid"])


class TestAlphaNetwork(unittest.TestCase):

    def test_network_forward(self):
        from neural.layer import Layer
        from neural.network import Network

        layer1 = Layer([
            Neuron([1, 1], activation="linear"),
            Neuron([2, 2], activation="linear")
        ])

        layer2 = Layer([
            Neuron([1, 1], activation="linear")
        ])

        network = Network([layer1, layer2])

        self.assertEqual(network.forward([2, 3]), [15.0])

    def test_network_layer_count(self):
        from neural.layer import Layer
        from neural.network import Network

        layer1 = Layer([Neuron([1])])
        layer2 = Layer([Neuron([2])])

        network = Network([layer1, layer2])

        self.assertEqual(network.layer_count(), 2)

    def test_empty_network(self):
        from neural.network import Network

        with self.assertRaises(ValueError):
            Network([])

    def test_invalid_layer(self):
        from neural.network import Network

        with self.assertRaises(TypeError):
            Network(["invalid"])


if __name__ == "__main__":
    unittest.main()
