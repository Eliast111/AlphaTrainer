import unittest

from learning.loss import mse, mse_derivative


class TestAlphaLoss(unittest.TestCase):

    def test_mse(self):
        result = mse([2, 4, 6], [1, 2, 3])
        self.assertAlmostEqual(result, 14 / 3)

    def test_mse_zero_error(self):
        self.assertEqual(mse([1, 2], [1, 2]), 0.0)

    def test_mse_derivative(self):
        result = mse_derivative([2, 4], [1, 2])
        self.assertEqual(result, [1.0, 2.0])

    def test_mse_invalid_sizes(self):
        with self.assertRaises(ValueError):
            mse([1, 2], [1])

    def test_mse_empty_lists(self):
        with self.assertRaises(ValueError):
            mse([], [])


if __name__ == "__main__":
    unittest.main()
