import unittest
from types import SimpleNamespace

from ai_runtime import AIRuntime, AIRuntimeError


class FakeRegistry:
    def __init__(self, models):
        self.models = models

    def get(self, name):
        if name not in self.models:
            raise KeyError(name)
        return self.models[name]


class FakeAdapter:
    def __init__(self, predict=False, generate=False):
        self.capabilities = SimpleNamespace(
            predict=predict,
            generate=generate,
        )


class FakeManager:
    def __init__(self, models, adapters):
        self.registry = FakeRegistry(models)
        self.adapters = adapters

    def adapter_for(self, name):
        if name not in self.adapters:
            raise KeyError(name)
        return self.adapters[name]


class DenseModel:
    def predict(self, inputs):
        return sum(inputs)

    def forward(self, inputs):
        return sum(inputs) * 2


class LanguageModel:
    def predict_next(self, tokens):
        return tokens[-1] + 1

    def generate(self, prompt="", **options):
        return prompt + " Alpha"


class TestAIRuntime(unittest.TestCase):

    def setUp(self):
        self.dense = DenseModel()
        self.language = LanguageModel()

        manager = FakeManager(
            {
                "dense": self.dense,
                "language": self.language,
            },
            {
                "dense": FakeAdapter(predict=True),
                "language": FakeAdapter(
                    predict=True, generate=True
                ),
            },
        )

        self.runtime = AIRuntime(manager)

    def test_dense_prediction(self):
        self.assertEqual(
            self.runtime.predict("dense", [2, 3]), 5
        )

    def test_transformer_prediction(self):
        self.assertEqual(
            self.runtime.predict("language", [1, 4]), 5
        )

    def test_text_generation(self):
        self.assertEqual(
            self.runtime.generate("language", "Ola"),
            "Ola Alpha",
        )

    def test_forward_execution(self):
        self.assertEqual(
            self.runtime.forward("dense", [2, 3]), 10
        )

    def test_rejects_unsupported_generation(self):
        with self.assertRaises(AIRuntimeError):
            self.runtime.generate("dense", "Ola")

    def test_unknown_operation(self):
        with self.assertRaises(AIRuntimeError):
            self.runtime.execute("dense", "train")

    def test_unknown_model(self):
        with self.assertRaises(KeyError):
            self.runtime.predict("inexistente", [1])

    def test_invalid_model_name(self):
        with self.assertRaises(ValueError):
            self.runtime.predict("", [1])


if __name__ == "__main__":
    unittest.main()
