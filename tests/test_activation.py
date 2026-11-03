import unittest

from alpha_math.activation import (
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
    linear,
    linear_derivative
)


class TestAlphaActivation(unittest.TestCase):

    def test_relu(self):
        self.assertEqual(relu(-2), 0)
        self.assertEqual(relu(3), 3)

    def test_relu_derivative(self):
        self.assertEqual(relu_derivative(-1), 0)
        self.assertEqual(relu_derivative(2), 1)

    def test_sigmoid(self):
        self.assertAlmostEqual(sigmoid(0), 0.5)

    def test_sigmoid_derivative(self):
        self.assertAlmostEqual(sigmoid_derivative(0), 0.25)

    def test_tanh(self):
        self.assertAlmostEqual(tanh(0), 0)

    def test_tanh_derivative(self):
        self.assertAlmostEqual(tanh_derivative(0), 1)

    def test_linear(self):
        self.assertEqual(linear(5), 5)
        self.assertEqual(linear_derivative(5), 1)


if __name__ == "__main__":
    unittest.main()
