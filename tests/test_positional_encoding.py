import unittest
import math

from alpha_math.tensor import Tensor
from transformer.positional_encoding import PositionalEncoding


class TestPositionalEncoding(unittest.TestCase):

    def setUp(self):
        self.encoding = PositionalEncoding(10, 4)

    def test_creation(self):
        self.assertEqual(self.encoding.max_length, 10)
        self.assertEqual(self.encoding.embedding_dim, 4)

    def test_encoding_shape(self):
        self.assertEqual(self.encoding.encoding.shape(), (10, 4))

    def test_first_position(self):
        position = self.encoding.get(0)
        self.assertEqual(position.data, [0.0, 1.0, 0.0, 1.0])

    def test_position_values(self):
        position = self.encoding.get(1)
        self.assertAlmostEqual(position.data[0], math.sin(1))
        self.assertAlmostEqual(position.data[1], math.cos(1))

    def test_get_position(self):
        self.assertEqual(self.encoding.get(3).shape(), (4,))

    def test_forward_shape(self):
        embeddings = Tensor([[0.1, 0.2, 0.3, 0.4]])
        result = self.encoding.forward(embeddings)
        self.assertEqual(result.shape(), (1, 4))

    def test_forward_adds_position(self):
        embeddings = Tensor([[0.0, 0.0, 0.0, 0.0]])
        result = self.encoding.forward(embeddings)

        self.assertEqual(result.data[0], self.encoding.get(0).data)

    def test_multiple_positions(self):
        embeddings = Tensor([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8]
        ])

        result = self.encoding.forward(embeddings)
        self.assertEqual(result.shape(), (2, 4))

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            PositionalEncoding(0, 4)

        with self.assertRaises(ValueError):
            PositionalEncoding(10, 0)

    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            self.encoding.forward([[0.1, 0.2]])

    def test_incompatible_dimension(self):
        with self.assertRaises(ValueError):
            self.encoding.forward(Tensor([[0.1, 0.2]]))

    def test_sequence_too_long(self):
        embeddings = Tensor([[0.0] * 4 for _ in range(11)])

        with self.assertRaises(ValueError):
            self.encoding.forward(embeddings)

    def test_invalid_position(self):
        with self.assertRaises(ValueError):
            self.encoding.get(10)

    def test_empty_sequence(self):
        with self.assertRaises(ValueError):
            self.encoding.forward(Tensor([]))


if __name__ == "__main__":
    unittest.main()
