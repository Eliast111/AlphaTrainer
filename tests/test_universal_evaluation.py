import unittest
from types import SimpleNamespace

from evaluation.universal import (
    UniversalEvaluationEngine,
    UniversalEvaluationError,
)


class FakeRegistry:
    def __init__(self, models):
        self.models = models

    def get(self, name):
        if name not in self.models:
            raise KeyError(name)
        return self.models[name]


class FakeAdapter:
    def __init__(self, architecture, evaluate=True):
        self.architecture = architecture
        self.capabilities = SimpleNamespace(evaluate=evaluate)


class FakeManager:
    def __init__(self, models, adapters):
        self.registry = FakeRegistry(models)
        self.adapters = adapters
        self.saved_metrics = {}

    def adapter_for(self, name):
        return self.adapters[name]

    def update_training(self, name, status=None, metrics=None,
                        dataset_id=None):
        if metrics is not None:
            self.saved_metrics.setdefault(name, []).append(metrics)


class DenseModel:
    def forward(self, inputs):
        return [inputs[0] * 2]


class TransformerModel:
    def evaluate(self, texts):
        return {
            "loss": 0.25,
            "perplexity": 1.28,
            "tokens": 10,
        }


class InvalidTransformer:
    def evaluate(self, texts):
        return None


class TestUniversalEvaluation(unittest.TestCase):

    def setUp(self):
        self.dense = DenseModel()
        self.transformer = TransformerModel()

        self.manager = FakeManager(
            {
                "dense": self.dense,
                "transformer": self.transformer,
            },
            {
                "dense": FakeAdapter("dense"),
                "transformer": FakeAdapter("transformer_decoder"),
            },
        )

        self.engine = UniversalEvaluationEngine(self.manager)

    def test_evaluates_dense_model(self):
        result = self.engine.evaluate(
            "dense",
            [([1], [2]), ([2], [4])],
        )

        self.assertEqual(result["architecture"], "dense")
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["metrics"]["mse"], 0)

    def test_evaluates_transformer(self):
        result = self.engine.evaluate(
            "transformer",
            ["ola alpha", "alpha technologies"],
        )

        self.assertEqual(
            result["architecture"], "transformer_decoder"
        )
        self.assertEqual(result["metrics"]["loss"], 0.25)
        self.assertEqual(result["samples"], 10)

    def test_saves_evaluation_metrics(self):
        self.engine.evaluate("transformer", ["ola alpha"])

        saved = self.manager.saved_metrics["transformer"]

        self.assertEqual(len(saved), 1)
        self.assertIn("evaluation", saved[0])

    def test_does_not_change_training_status(self):
        status = "created"

        self.engine.evaluate("dense", [([1], [2])])

        self.assertEqual(status, "created")

    def test_rejects_unsupported_evaluation(self):
        manager = FakeManager(
            {"model": DenseModel()},
            {"model": FakeAdapter("dense", evaluate=False)},
        )
        engine = UniversalEvaluationEngine(manager)

        with self.assertRaises(UniversalEvaluationError):
            engine.evaluate("model", [([1], [2])])

    def test_rejects_invalid_transformer_result(self):
        manager = FakeManager(
            {"model": InvalidTransformer()},
            {"model": FakeAdapter("transformer_decoder")},
        )
        engine = UniversalEvaluationEngine(manager)

        with self.assertRaises(UniversalEvaluationError):
            engine.evaluate("model", ["ola"])

    def test_rejects_invalid_name(self):
        with self.assertRaises(ValueError):
            self.engine.evaluate("", [])

    def test_unknown_model(self):
        with self.assertRaises(KeyError):
            self.engine.evaluate("inexistente", [])


if __name__ == "__main__":
    unittest.main()
