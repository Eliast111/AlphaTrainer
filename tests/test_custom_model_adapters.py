import unittest

from core import AlphaCore
from models.contracts import ModelCapabilities


class CustomModel:
    def fit(self, data, epochs=1, **options):
        return {"epochs_run": epochs, "loss": 0.1}

    def evaluate(self, data):
        return {"loss": 0.1, "samples": len(data)}

    def predict(self, inputs):
        return sum(inputs) + self.bias

    def __init__(self, bias=0):
        self.bias = bias

    def forward(self, inputs):
        return self.predict(inputs) * 2


class CustomModelAdapter:
    model_type = "custom"
    architecture = "custom_test"
    capabilities = ModelCapabilities(
        train=True,
        evaluate=True,
        predict=True,
        forward=True,
    )

    def validate(self, model):
        if not isinstance(model, CustomModel):
            raise TypeError("Modelo personalizado invalido.")

    def export_state(self, model):
        return {"bias": model.bias}

    def restore_state(self, state):
        return CustomModel(bias=state["bias"])


class InvalidAdapter:
    model_type = "custom"
    architecture = "invalid"
    capabilities = {"predict": True}

    def validate(self, model):
        pass


class CustomModelAdapterTests(unittest.TestCase):
    def setUp(self):
        self.core = AlphaCore()

    def test_legacy_registration_keeps_capabilities_disabled(self):
        self.core.register_model("legacy", object())
        capabilities = self.core.model_capabilities("legacy")
        self.assertFalse(any(capabilities.values()))

    def test_custom_adapter_registration(self):
        model = CustomModel()
        adapter = CustomModelAdapter()

        self.core.register_model(
            "custom", model, adapter=adapter
        )

        self.assertIs(self.core.get_model("custom"), model)
        self.assertEqual(
            self.core.model_metadata("custom")["architecture"],
            "custom_test",
        )
        self.assertTrue(
            self.core.model_capabilities("custom")["train"]
        )

    def test_custom_model_runtime_prediction(self):
        self.core.register_model(
            "custom", CustomModel(), adapter=CustomModelAdapter()
        )
        self.assertEqual(
            self.core.runtime_predict("custom", [2, 3]), 5
        )
        self.assertEqual(
            self.core.runtime_forward("custom", [2, 3]), 10
        )

    def test_custom_model_training_and_evaluation(self):
        self.core.register_model(
            "custom", CustomModel(), adapter=CustomModelAdapter()
        )

        result = self.core.train_universal(
            "custom", [1, 2], epochs=3
        )
        self.assertEqual(result["epochs_run"], 3)

        evaluation = self.core.evaluate_universal(
            "custom", [1, 2]
        )
        self.assertEqual(evaluation["metrics"]["loss"], 0.1)

    def test_custom_adapter_state_export_and_restore(self):
        adapter = CustomModelAdapter()
        model = CustomModel(bias=7)
        self.core.register_model("custom", model, adapter=adapter)

        payload = self.core.model_manager.export_custom_state("custom")

        restored = AlphaCore()
        restored.model_manager.register_adapter(adapter)
        restored.model_manager.restore_custom_model("recovered", payload)

        self.assertEqual(
            restored.runtime_predict("recovered", [2, 3]), 12
        )

    def test_custom_restore_rejects_unregistered_adapter(self):
        adapter = CustomModelAdapter()
        self.core.register_model(
            "custom", CustomModel(bias=4), adapter=adapter
        )
        payload = self.core.model_manager.export_custom_state("custom")

        restored = AlphaCore()
        with self.assertRaises((KeyError, ValueError)):
            restored.model_manager.restore_custom_model("recovered", payload)

    def test_invalid_adapter_rejected(self):
        with self.assertRaises((TypeError, ValueError)):
            self.core.register_model(
                "invalid", CustomModel(), adapter=InvalidAdapter()
            )


if __name__ == "__main__":
    unittest.main()
