import unittest

from learning.gradients import numerical_gradient


class TestAlphaGradients(unittest.TestCase):

    def test_quadratic_gradient(self):
        def function(values):
            return values[0] ** 2 + values[1] ** 2

        result = numerical_gradient(function, [3.0, 4.0])

        self.assertAlmostEqual(result[0], 6.0, places=4)
        self.assertAlmostEqual(result[1], 8.0, places=4)

    def test_linear_gradient(self):
        def function(values):
            return 5 * values[0] + 3 * values[1]

        result = numerical_gradient(function, [2.0, 4.0])

        self.assertAlmostEqual(result[0], 5.0, places=4)
        self.assertAlmostEqual(result[1], 3.0, places=4)

    def test_empty_parameters(self):
        result = numerical_gradient(lambda values: 10, [])
        self.assertEqual(result, [])

    def test_invalid_epsilon(self):
        with self.assertRaises(ValueError):
            numerical_gradient(lambda values: values[0] ** 2, [2], 0)


if __name__ == "__main__":
    unittest.main()
