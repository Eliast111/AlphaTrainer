import unittest

from models.factory import create_model


def weights_of(model):
    return [n.weights for layer in model.layers for n in layer.neurons]


class TestFactoryInit(unittest.TestCase):

    def test_random_init_breaks_symmetry(self):
        model = create_model([2, 4, 1])
        first = model.layers[0].neurons
        self.assertNotEqual(first[0].weights, first[1].weights)

    def test_seed_is_reproducible(self):
        a = create_model([2, 4, 1], seed=7)
        b = create_model([2, 4, 1], seed=7)
        c = create_model([2, 4, 1], seed=8)
        self.assertEqual(weights_of(a), weights_of(b))
        self.assertNotEqual(weights_of(a), weights_of(c))

    def test_constant_init_matches_old_behavior(self):
        model = create_model([2, 2, 1], init="constant")
        for weights in weights_of(model):
            self.assertTrue(all(w == 0.01 for w in weights))

    def test_output_activation(self):
        model = create_model(
            [2, 3, 1], activation="tanh", output_activation="sigmoid"
        )
        self.assertEqual(model.layers[0].neurons[0].activation_name, "tanh")
        self.assertEqual(model.layers[1].neurons[0].activation_name, "sigmoid")

    def test_invalid_activation(self):
        with self.assertRaises(ValueError):
            create_model([2, 1], activation="nope")
        with self.assertRaises(ValueError):
            create_model([2, 1], output_activation="nope")

    def test_invalid_init(self):
        with self.assertRaises(ValueError):
            create_model([2, 1], init="nope")

    def test_network_helpers(self):
        model = create_model([2, 3, 1])
        self.assertEqual(model.layer_sizes(), [2, 3, 1])
        self.assertEqual(model.parameter_count(), 3 * 3 + 4)
        self.assertEqual(model.input_size(), 2)
        self.assertEqual(model.output_size(), 1)

    def test_leaky_relu(self):
        model = create_model([1, 1], activation="leaky_relu")
        neuron = model.layers[0].neurons[0]
        neuron.weights = [1.0]
        self.assertAlmostEqual(neuron.forward([-2.0]), -0.02)


if __name__ == "__main__":
    unittest.main()
