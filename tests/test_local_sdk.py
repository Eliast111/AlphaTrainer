import unittest

from sdk.local import LocalLearningSDK, LocalSDKError


class FakeCore:
    def __init__(self):
        self.calls = []
        self.caps = {
            "dense": {"train": True, "predict": True, "forward": True,
                      "evaluate": True, "resume": True, "generate": False},
            "text": {"train": True, "predict": True, "forward": True,
                     "evaluate": True, "resume": False, "generate": True},
        }

    def create_model(self, *args, **kwargs): return (args, kwargs)
    def create_language_model(self, *args, **kwargs): return (args, kwargs)
    def register_model(self, *args, **kwargs):
        self.registration = (args, kwargs)
        return args
    def list_models(self): return ["dense", "text"]
    def model_metadata(self, name): return {"name": name}
    def model_capabilities(self, name): return self.caps[name]
    def train_universal(self, *args, **kwargs):
        self.calls.append((args, kwargs)); return {"ok": True}
    def runtime_predict(self, name, inputs): return [sum(inputs)]
    def runtime_forward(self, name, inputs): return [sum(inputs) * 2]
    def runtime_generate(self, name, prompt="", **options): return prompt + "!"
    def evaluate_universal(self, name, dataset): return {"mse": 0.0}


class LocalSDKTests(unittest.TestCase):
    def setUp(self):
        self.core = FakeCore()
        self.sdk = LocalLearningSDK(self.core)

    def test_list_and_metadata(self):
        self.assertEqual(self.sdk.list_models(), ["dense", "text"])
        self.assertEqual(self.sdk.metadata("dense"), {"name": "dense"})

    def test_train_dense(self):
        self.assertEqual(self.sdk.train("dense", [], epochs=3), {"ok": True})

    def test_resume_rejected_for_unsupported_model(self):
        with self.assertRaises(LocalSDKError):
            self.sdk.train("text", [], resume=True)

    def test_predict_forward_generate_and_evaluate(self):
        self.assertEqual(self.sdk.predict("dense", [2, 3]), [5])
        self.assertEqual(self.sdk.forward("dense", [2, 3]), [10])
        self.assertEqual(self.sdk.generate("text", "Alpha"), "Alpha!")
        self.assertEqual(self.sdk.evaluate("dense", []), {"mse": 0.0})

    def test_generation_rejected_when_unsupported(self):
        with self.assertRaises(LocalSDKError):
            self.sdk.generate("dense", "Alpha")

    def test_register_custom_adapter_forwarded(self):
        model = object()
        adapter = object()

        result = self.sdk.register_model(
            "custom", model, adapter=adapter
        )

        self.assertEqual(result, ("custom", model))
        args, kwargs = self.core.registration
        self.assertEqual(args, ("custom", model))
        self.assertIs(kwargs["adapter"], adapter)

    def test_invalid_core_rejected(self):
        with self.assertRaises(TypeError):
            LocalLearningSDK(object())


if __name__ == "__main__":
    unittest.main()
