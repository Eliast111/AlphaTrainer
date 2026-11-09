import unittest

from learning.language_gradients import output_layer_gradients


class TestLanguageGradients(unittest.TestCase):

    def test_output_gradients(self):
        result = output_layer_gradients(
            [[1.0, 0.5]],
            [[0.8, 0.2]],
            [0],
            [[0.1, 0.2], [0.3, 0.4]]
        )
        self.assertAlmostEqual(result["weights"][0][0], -0.2)
        self.assertAlmostEqual(result["weights"][0][1], 0.2)
        self.assertAlmostEqual(result["bias"][0], -0.2)
        self.assertEqual(len(result["hidden"]), 1)

    def test_rejects_invalid_target(self):
        with self.assertRaises(ValueError):
            output_layer_gradients(
                [[1.0]], [[1.0, 0.0]], [2], [[0.1, 0.2]]
            )

    def test_rejects_incompatible_dimensions(self):
        with self.assertRaises(ValueError):
            output_layer_gradients(
                [[1.0, 2.0]], [[0.5, 0.5]], [0], [[0.1, 0.2]]
            )


if __name__ == "__main__":
    unittest.main()
