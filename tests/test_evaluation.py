import unittest
import math

from evaluation.metrics import mse, mae, rmse, evaluate


class TestAlphaEvaluation(unittest.TestCase):

    def setUp(self):
        self.predictions = [2.0, 4.0, 6.0]
        self.targets = [1.0, 5.0, 5.0]

    def test_mse(self):
        self.assertAlmostEqual(
            mse(self.predictions, self.targets),
            1.0
        )

    def test_mae(self):
        self.assertAlmostEqual(
            mae(self.predictions, self.targets),
            1.0
        )

    def test_rmse(self):
        self.assertAlmostEqual(
            rmse(self.predictions, self.targets),
            1.0
        )

    def test_evaluate(self):
        result = evaluate(self.predictions, self.targets)

        self.assertEqual(
            set(result.keys()),
            {"mse", "mae", "rmse"}
        )

        self.assertAlmostEqual(result["mse"], 1.0)
        self.assertAlmostEqual(result["mae"], 1.0)
        self.assertAlmostEqual(result["rmse"], 1.0)

    def test_zero_error(self):
        values = [1.0, 2.0, 3.0]

        self.assertEqual(mse(values, values), 0.0)
        self.assertEqual(mae(values, values), 0.0)
        self.assertEqual(rmse(values, values), 0.0)

    def test_empty_predictions(self):
        with self.assertRaises(ValueError):
            mse([], [])

    def test_different_sizes(self):
        with self.assertRaises(ValueError):
            mae([1.0, 2.0], [1.0])

    def test_non_numeric_values(self):
        with self.assertRaises(ValueError):
            mse(["abc"], [1.0])

    def test_non_finite_values(self):
        with self.assertRaises(ValueError):
            evaluate([float("nan")], [1.0])

    def test_negative_values(self):
        self.assertAlmostEqual(
            mae([-2.0, -4.0], [-1.0, -5.0]),
            1.0
        )


if __name__ == "__main__":
    unittest.main()
