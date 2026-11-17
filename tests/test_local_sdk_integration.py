import unittest

from core import AlphaCore
from data.dataset import Dataset
from learning.optimizer import GradientDescent
from sdk.local import LocalLearningSDK


class LocalSDKIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.core = AlphaCore()
        self.sdk = LocalLearningSDK(self.core)

    def test_dense_model_complete_workflow(self):
        self.sdk.create_dense_model(
            "dense_integracao", [1, 1], activation="linear"
        )
        dataset = Dataset([
            ([1.0], [2.0]),
            ([2.0], [4.0]),
        ])

        capabilities = self.sdk.capabilities("dense_integracao")
        self.assertTrue(capabilities["train"])
        self.assertTrue(capabilities["predict"])
        self.assertTrue(capabilities["forward"])
        self.assertTrue(capabilities["evaluate"])

        training = self.sdk.train(
            "dense_integracao",
            dataset,
            optimizer=GradientDescent(learning_rate=0.01),
            epochs=2,
        )
        self.assertEqual(training["epochs"], 2)
        self.assertEqual(len(training["history"]), 2)

        prediction = self.sdk.predict("dense_integracao", [1.5])
        self.assertEqual(len(prediction), 1)

        output = self.sdk.forward("dense_integracao", [1.5])
        self.assertIsNotNone(output)

        evaluation = self.sdk.evaluate("dense_integracao", dataset)
        self.assertIsInstance(evaluation, dict)
        self.assertTrue(evaluation)

    def test_transformer_complete_workflow(self):
        texts = ["ola mundo .", "ola alpha ."]
        self.sdk.create_language_model(
            "texto_integracao",
            texts,
            embedding_dim=4,
            hidden_dim=4,
            num_layers=1,
            max_length=8,
        )

        capabilities = self.sdk.capabilities("texto_integracao")
        self.assertTrue(capabilities["train"])
        self.assertTrue(capabilities["predict"])
        self.assertTrue(capabilities["forward"])
        self.assertTrue(capabilities["generate"])
        self.assertTrue(capabilities["evaluate"])
        self.assertFalse(capabilities["resume"])

        training = self.sdk.train(
            "texto_integracao",
            texts,
            epochs=2,
            batch_size=2,
            learning_rate=0.01,
            seed=7,
        )
        self.assertEqual(training["epochs_run"], 2)
        self.assertEqual(len(training["history"]), 2)

        model = self.core.get_model("texto_integracao")
        token_ids = model.encode_text("ola")

        prediction = self.sdk.predict("texto_integracao", token_ids)
        self.assertIsInstance(prediction, int)

        output = self.sdk.forward("texto_integracao", token_ids)
        self.assertIsNotNone(output)

        generated = self.sdk.generate(
            "texto_integracao", "ola", max_new_tokens=2, seed=7
        )
        self.assertIsInstance(generated, str)

        evaluation = self.sdk.evaluate("texto_integracao", texts)
        self.assertIn("metrics", evaluation)
        self.assertIn("loss", evaluation["metrics"])
        self.assertIn("perplexity", evaluation["metrics"])
        self.assertGreater(evaluation["metrics"]["tokens"], 0)


if __name__ == "__main__":
    unittest.main()
