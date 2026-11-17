import unittest

from language.vocabulary import Vocabulary
from models.ai_factory import AIModelFactory
from neural.network import Network
from runtime import AlphaRuntime
from transformer.language_model import LanguageModel


class TestAIModelFactory(unittest.TestCase):
    def setUp(self):
        self.factory = AIModelFactory()

    def test_create_feedforward_model(self):
        model = self.factory.create("feedforward", layer_sizes=[2, 1])
        self.assertIsInstance(model, Network)

    def test_create_language_model(self):
        vocabulary = Vocabulary().build([["ola", "mundo"]])
        model = self.factory.create(
            "language_model", vocabulary=vocabulary, embedding_dim=4
        )
        self.assertIsInstance(model, LanguageModel)

    def test_unknown_type(self):
        with self.assertRaises(ValueError):
            self.factory.create("unknown")

    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            self.factory.create(" ")

    def test_register_custom_builder(self):
        self.factory.register("custom", lambda value: {"value": value})
        self.assertEqual(self.factory.create("custom", value=3), {"value": 3})

    def test_duplicate_builder(self):
        with self.assertRaises(ValueError):
            self.factory.register("feedforward", lambda: None)

    def test_invalid_builder(self):
        with self.assertRaises(TypeError):
            self.factory.register("custom", None)

    def test_model_types_are_read_only_snapshot(self):
        model_types = self.factory.model_types
        self.factory.register("custom", lambda: object())
        self.assertNotIn("custom", model_types)
        self.assertIn("custom", self.factory.model_types)


class TestAlphaRuntime(unittest.TestCase):
    def test_run_feedforward_model(self):
        model = AIModelFactory().create(
            "feedforward", layer_sizes=[2, 1], activation="linear"
        )
        self.assertEqual(len(AlphaRuntime.run(model, [1.0, 2.0])), 1)

    def test_run_rejects_models_without_forward(self):
        with self.assertRaises(TypeError):
            AlphaRuntime.run(object(), [])

    def test_predict_next_rejects_incompatible_model(self):
        with self.assertRaises(TypeError):
            AlphaRuntime.predict_next(object(), [1])

    def test_generate_language_model_tokens(self):
        vocabulary = Vocabulary().build([["ola", "mundo"]])
        model = AIModelFactory().create(
            "language_model", vocabulary=vocabulary, embedding_dim=4,
            hidden_dim=4, max_length=8, seed=1
        )
        result = AlphaRuntime().generate(
            model, [vocabulary.get_id("ola")], max_new_tokens=2
        )
        self.assertEqual(len(result), 3)

    def test_generate_stops_at_eos(self):
        class SequenceModel:
            def predict_next(self, token_ids):
                return 9

        result = AlphaRuntime().generate(
            SequenceModel(), [1], max_new_tokens=5, eos_id=9
        )
        self.assertEqual(result, [1, 9])

    def test_generate_validates_limit(self):
        with self.assertRaises(ValueError):
            AlphaRuntime().generate(object(), [1], max_new_tokens=-1)

    def test_generate_validates_token_context(self):
        with self.assertRaises(ValueError):
            AlphaRuntime().generate(object(), [], max_new_tokens=1)


if __name__ == "__main__":
    unittest.main()
