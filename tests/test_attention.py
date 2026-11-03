import unittest

from alpha_math.tensor import Tensor
from transformer.attention import SelfAttention


class TestSelfAttention(unittest.TestCase):

    def setUp(self):
        self.attention = SelfAttention()

        self.query = Tensor([[1.0, 0.0], [0.0, 1.0]])
        self.key = Tensor([[1.0, 0.0], [0.0, 1.0]])
        self.value = Tensor([[2.0, 0.0], [0.0, 2.0]])

    def test_output_shape(self):
        output, scores = self.attention.forward(
            self.query, self.key, self.value
        )
        self.assertEqual(output.shape(), (2, 2))
        self.assertEqual(scores.shape(), (2, 2))

    def test_attention_probabilities_sum_to_one(self):
        _, scores = self.attention.forward(
            self.query, self.key, self.value
        )

        for row in scores.data:
            self.assertAlmostEqual(sum(row), 1.0)

    def test_attention_probabilities_are_positive(self):
        _, scores = self.attention.forward(
            self.query, self.key, self.value
        )

        for row in scores.data:
            for value in row:
                self.assertGreater(value, 0.0)

    def test_softmax(self):
        result = self.attention.softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(result), 1.0)
        self.assertLess(result[0], result[1])
        self.assertLess(result[1], result[2])

    def test_softmax_empty(self):
        with self.assertRaises(ValueError):
            self.attention.softmax([])

    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            self.attention.forward([], self.key, self.value)

    def test_causal_attention(self):
        causal = SelfAttention(causal=True)

        _, scores = causal.forward(
            self.query, self.key, self.value
        )

        self.assertEqual(scores.data[0][1], 0.0)
        self.assertAlmostEqual(sum(scores.data[1]), 1.0)

    def test_causal_configuration(self):
        with self.assertRaises(TypeError):
            SelfAttention(causal="yes")

    def test_incompatible_dimensions(self):
        bad_key = Tensor([[1.0, 2.0, 3.0]])

        with self.assertRaises(ValueError):
            self.attention.forward(
                self.query, bad_key, self.value
            )

    def test_empty_matrices(self):
        with self.assertRaises(ValueError):
            self.attention.forward(
                Tensor([]), Tensor([]), Tensor([])
            )


if __name__ == "__main__":
    unittest.main()
