import unittest

from evaluation.report import generate_report
from data.dataset import Dataset
from neural.neuron import Neuron
from neural.layer import Layer
from neural.network import Network


class TestAlphaReport(unittest.TestCase):

    def setUp(self):
        self.dataset = Dataset([
            ([1.0], [2.0]),
            ([2.0], [4.0]),
            ([3.0], [6.0]),
        ])

        self.model = Network([
            Layer([
                Neuron([2.0], bias=0.0, activation="linear")
            ])
        ])

    def test_report_generation(self):
        report = generate_report(self.model, self.dataset)

        self.assertIn("summary", report)
        self.assertIn("sample_reports", report)

    def test_report_sample_count(self):
        report = generate_report(self.model, self.dataset)

        self.assertEqual(report["summary"]["samples"], 3)

    def test_report_metrics(self):
        report = generate_report(self.model, self.dataset)

        self.assertAlmostEqual(report["summary"]["mse"], 0.0)
        self.assertAlmostEqual(report["summary"]["mae"], 0.0)
        self.assertAlmostEqual(report["summary"]["rmse"], 0.0)

    def test_individual_sample_reports(self):
        report = generate_report(self.model, self.dataset)

        self.assertEqual(len(report["sample_reports"]), 3)

    def test_sample_absolute_errors(self):
        report = generate_report(self.model, self.dataset)

        first = report["sample_reports"][0]

        self.assertEqual(first["absolute_errors"], [0.0])

    def test_worst_sample(self):
        dataset = Dataset([
            ([1.0], [2.0]),
            ([2.0], [5.0]),
            ([3.0], [6.0]),
        ])

        report = generate_report(self.model, dataset)

        self.assertEqual(report["worst_sample"], 2)
        self.assertAlmostEqual(report["worst_sample_mae"], 1.0)


if __name__ == "__main__":
    unittest.main()
