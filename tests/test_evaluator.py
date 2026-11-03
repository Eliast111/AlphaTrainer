import unittest

from evaluation.evaluator import evaluate_model, compare_models
from data.dataset import Dataset
from neural.neuron import Neuron
from neural.layer import Layer
from neural.network import Network


class TestAlphaEvaluator(unittest.TestCase):

    def setUp(self):
        self.dataset = Dataset([
            ([1.0], [2.0]),
            ([2.0], [4.0]),
            ([3.0], [6.0]),
        ])

        self.good_model = Network([
            Layer([
                Neuron([2.0], bias=0.0, activation="linear")
            ])
        ])

        self.bad_model = Network([
            Layer([
                Neuron([1.0], bias=0.0, activation="linear")
            ])
        ])

    def test_evaluate_model(self):
        result = evaluate_model(self.good_model, self.dataset)

        self.assertEqual(result["samples"], 3)
        self.assertAlmostEqual(result["metrics"]["mse"], 0.0)

    def test_predictions_returned(self):
        result = evaluate_model(self.good_model, self.dataset)

        self.assertEqual(
            result["predictions"],
            [2.0, 4.0, 6.0]
        )

    def test_targets_returned(self):
        result = evaluate_model(self.good_model, self.dataset)

        self.assertEqual(
            result["targets"],
            [2.0, 4.0, 6.0]
        )

    def test_compare_models(self):
        result = compare_models(
            {
                "Alpha A": self.good_model,
                "Alpha B": self.bad_model,
            },
            self.dataset
        )

        self.assertEqual(result["best_model"], "Alpha A")

    def test_model_ranking(self):
        result = compare_models(
            {
                "Alpha A": self.bad_model,
                "Alpha B": self.good_model,
            },
            self.dataset
        )

        self.assertEqual(
            result["ranking"][0],
            "Alpha B"
        )

    def test_invalid_model(self):
        with self.assertRaises(TypeError):
            evaluate_model(object(), self.dataset)

    def test_empty_models(self):
        with self.assertRaises(ValueError):
            compare_models({}, self.dataset)

    def test_empty_dataset(self):
        with self.assertRaises(ValueError):
            evaluate_model(self.good_model, [])

    def test_prediction_size_mismatch(self):
        class BadModel:
            def forward(self, inputs):
                return [1.0, 2.0]

        with self.assertRaises(ValueError):
            evaluate_model(BadModel(), self.dataset)


if __name__ == "__main__":
    unittest.main()
