import unittest

from learning.language_loss import cross_entropy, cross_entropy_derivative


class TestLanguageLoss(unittest.TestCase):

    def test_cross_entropy(self):
        probabilities = [[0.8, 0.2], [0.1, 0.9]]
        targets = [0, 1]
        result = cross_entropy(probabilities, targets)
        self.assertAlmostEqual(result, 0.164252033486018)

    def test_cross_entropy_derivative(self):
        probabilities = [[0.8, 0.2], [0.1, 0.9]]
        targets = [0, 1]
        result = cross_entropy_derivative(probabilities, targets)
        self.assertAlmostEqual(result[0][0], -0.625)
        self.assertAlmostEqual(result[1][1], -0.5555555555555556)

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            cross_entropy([], [])

    def test_different_lengths(self):
        with self.assertRaises(ValueError):
            cross_entropy([[0.5, 0.5]], [0, 1])

    def test_invalid_target(self):
        with self.assertRaises(ValueError):
            cross_entropy([[0.5, 0.5]], [2])

    def test_invalid_probability(self):
        with self.assertRaises(ValueError):
            cross_entropy([[1.2, -0.2]], [0])


if __name__ == "__main__":
    unittest.main()
