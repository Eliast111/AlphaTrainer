import unittest

from alpha_math.tensor import Tensor
from language.embedding import Embedding


class TestEmbedding(unittest.TestCase):

    def setUp(self):
        self.embedding = Embedding(10, 4, seed=42)

    def test_creation(self):
        self.assertEqual(self.embedding.vocab_size, 10)
        self.assertEqual(self.embedding.embedding_dim, 4)

    def test_weights_are_tensor(self):
        self.assertIsInstance(self.embedding.weights, Tensor)

    def test_weight_matrix_shape(self):
        self.assertEqual(self.embedding.weights.shape(), (10, 4))

    def test_get_embedding(self):
        vector = self.embedding.get(2)
        self.assertIsInstance(vector, Tensor)
        self.assertEqual(vector.shape(), (4,))

    def test_forward(self):
        result = self.embedding.forward([1, 3, 5])
        self.assertIsInstance(result, Tensor)
        self.assertEqual(result.shape(), (3, 4))

    def test_deterministic_initialization(self):
        other = Embedding(10, 4, seed=42)
        self.assertEqual(
            self.embedding.weights.data,
            other.weights.data
        )

    def test_different_seeds(self):
        other = Embedding(10, 4, seed=99)
        self.assertNotEqual(
            self.embedding.weights.data,
            other.weights.data
        )

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            Embedding(0, 4)

        with self.assertRaises(ValueError):
            Embedding(10, 0)

    def test_invalid_token_id(self):
        with self.assertRaises(ValueError):
            self.embedding.get(10)

        with self.assertRaises(ValueError):
            self.embedding.get(-1)

    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            self.embedding.get("1")

        with self.assertRaises(TypeError):
            self.embedding.forward("123")


if __name__ == "__main__":
    unittest.main()
