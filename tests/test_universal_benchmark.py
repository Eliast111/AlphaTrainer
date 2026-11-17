import unittest
from types import SimpleNamespace

from evaluation.benchmark import (
    UniversalBenchmarkEngine,
    BenchmarkError,
)


class FakeRegistry:
    def __init__(self, models):
        self.models = models

    def get(self, name):
        return self.models[name]


class FakeAdapter:
    def __init__(self):
        self.architecture = "dense"
        self.capabilities = SimpleNamespace(evaluate=True)


class FakeModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def forward(self, inputs):
        return [self.prediction]


class FakeManager:
    def __init__(self):
        self.registry = FakeRegistry({
            "modelo_a": FakeModel(2),
            "modelo_b": FakeModel(4),
        })

    def adapter_for(self, name):
        return FakeAdapter()

    def update_training(self, name, status=None, metrics=None,
                        dataset_id=None):
        pass


class TestUniversalBenchmark(unittest.TestCase):

    def setUp(self):
        self.engine = UniversalBenchmarkEngine(FakeManager())
        self.data = {
            "modelo_a": [([1], [2])],
            "modelo_b": [([1], [2])],
        }

    def test_ranks_models_by_mse(self):
        result = self.engine.run(
            self.data, metric="mse", direction="min"
        )

        self.assertEqual(result["best_model"], "modelo_a")
        self.assertEqual(result["ranking"][0], "modelo_a")

    def test_ranks_in_reverse_direction(self):
        result = self.engine.run(
            self.data, metric="mse", direction="max"
        )

        self.assertEqual(result["best_model"], "modelo_b")

    def test_returns_metric_and_value(self):
        result = self.engine.run(self.data)

        self.assertEqual(result["metric"], "mse")
        self.assertEqual(result["best_value"], 0)

    def test_requires_two_models(self):
        with self.assertRaises(ValueError):
            self.engine.run({"modelo_a": self.data["modelo_a"]})

    def test_rejects_invalid_direction(self):
        with self.assertRaises(ValueError):
            self.engine.run(self.data, direction="middle")

    def test_rejects_missing_metric(self):
        with self.assertRaises(BenchmarkError):
            self.engine.run(self.data, metric="perplexity")

    def test_rejects_empty_metric(self):
        with self.assertRaises(ValueError):
            self.engine.run(self.data, metric="")

    def test_rejects_non_finite_metric(self):
        class InvalidModel:
            def forward(self, inputs):
                return [float("nan")]

        manager = FakeManager()
        manager.registry.models["modelo_a"] = InvalidModel()

        engine = UniversalBenchmarkEngine(manager)

        with self.assertRaises(ValueError):
            engine.run(self.data)


if __name__ == "__main__":
    unittest.main()
