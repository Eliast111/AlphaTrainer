import unittest

from models.factory import create_model
from neural.network import Network


class TestAlphaModelFactory(unittest.TestCase):

    def test_model_creation(self):
        model = create_model([2, 3, 1])
        self.assertIsInstance(model, Network)

    def test_layer_count(self):
        model = create_model([2, 3, 1])
        self.assertEqual(len(model.layers), 2)

    def test_first_layer_neurons(self):
        model = create_model([2, 3, 1])
        self.assertEqual(len(model.layers[0].neurons), 3)

    def test_second_layer_neurons(self):
        model = create_model([2, 3, 1])
        self.assertEqual(len(model.layers[1].neurons), 1)

    def test_invalid_layer_sizes(self):
        with self.assertRaises(ValueError):
            create_model([2])

    def test_invalid_neuron_count(self):
        with self.assertRaises(ValueError):
            create_model([2, 0])

    def test_invalid_input_type(self):
        with self.assertRaises(ValueError):
            create_model("2,3,1")

    def test_custom_activation(self):
        model = create_model([2, 1], activation="sigmoid")
        self.assertEqual(
            model.layers[0].neurons[0].activation_name,
            "sigmoid"
        )


if __name__ == "__main__":
    unittest.main()
