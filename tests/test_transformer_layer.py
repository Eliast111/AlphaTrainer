import unittest

from alpha_math.tensor import Tensor
from transformer.layer import TransformerLayer


class TestTransformerLayer(unittest.TestCase):

    def setUp(self):
        self.layer = TransformerLayer(
            embedding_dim=4,
            hidden_dim=8
        )
        self.inputs = Tensor([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8]
        ])

    def test_creation(self):
        self.assertEqual(self.layer.embedding_dim, 4)
        self.assertEqual(self.layer.hidden_dim, 8)

    def test_forward_shape(self):
        result = self.layer.forward(self.inputs)
        self.assertEqual(result.shape(), (2, 4))

    def test_forward_returns_tensor(self):
        result = self.layer.forward(self.inputs)
        self.assertIsInstance(result, Tensor)

    def test_forward_with_attention(self):
        result, scores = self.layer.forward_with_attention(self.inputs)
        self.assertEqual(result.shape(), (2, 4))
        self.assertEqual(scores.shape(), (2, 2))

    def test_attention_probabilities(self):
        _, scores = self.layer.forward_with_attention(self.inputs)

        for row in scores.data:
            self.assertAlmostEqual(sum(row), 1.0)

    def test_different_sequence_lengths(self):
        inputs = Tensor([[0.1, 0.2, 0.3, 0.4]])
        result = self.layer.forward(inputs)
        self.assertEqual(result.shape(), (1, 4))

    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            self.layer.forward([[0.1, 0.2, 0.3, 0.4]])

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            self.layer.forward(Tensor([[0.1, 0.2]]))

    def test_empty_sequence(self):
        with self.assertRaises(ValueError):
            self.layer.forward(Tensor([]))

    def test_invalid_embedding_dim(self):
        with self.assertRaises(ValueError):
            TransformerLayer(0, 8)

    def test_invalid_hidden_dim(self):
        with self.assertRaises(ValueError):
            TransformerLayer(4, 0)

    def test_invalid_causal_type(self):
        with self.assertRaises(TypeError):
            TransformerLayer(4, 8, causal="yes")

    def test_invalid_epsilon(self):
        with self.assertRaises(ValueError):
            TransformerLayer(4, 8, epsilon=0)


if __name__ == "__main__":
    unittest.main()
