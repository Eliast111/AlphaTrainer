import unittest

from core import AlphaCore
from data.dataset import Dataset
from language.vocabulary import Vocabulary
from learning.optimizer import GradientDescent
from neural.network import Network
from transformer.language_model import LanguageModel


class TestAlphaCore(unittest.TestCase):

    def setUp(self):
        self.alpha = AlphaCore()

    def test_initial_model_count(self):
        self.assertEqual(self.alpha.model_count(), 0)

    def test_create_model(self):
        model = self.alpha.create_model("modelo_teste", [2, 3, 1])
        self.assertIsNotNone(model)
        self.assertEqual(self.alpha.model_count(), 1)

    def test_get_model(self):
        created = self.alpha.create_model("modelo_teste", [2, 1])
        retrieved = self.alpha.get_model("modelo_teste")
        self.assertIs(created, retrieved)

    def test_list_models(self):
        self.alpha.create_model("modelo_a", [2, 1])
        self.alpha.create_model("modelo_b", [3, 1])
        self.assertEqual(
            self.alpha.list_models(),
            ["modelo_a", "modelo_b"]
        )

    def test_remove_model(self):
        self.alpha.create_model("modelo_teste", [2, 1])
        removed = self.alpha.remove_model("modelo_teste")
        self.assertIsNotNone(removed)
        self.assertEqual(self.alpha.model_count(), 0)

    def test_register_model(self):
        model = object()
        registered = self.alpha.register_model("modelo_externo", model)
        self.assertIs(model, registered)
        self.assertIs(model, self.alpha.get_model("modelo_externo"))

    def test_invalid_model_name(self):
        with self.assertRaises(ValueError):
            self.alpha.create_model("", [2, 1])

    def test_invalid_layer_sizes(self):
        with self.assertRaises(ValueError):
            self.alpha.create_model("modelo_invalido", [2])

    def test_create_model_by_type(self):
        model = self.alpha.create_model_by_type(
            "modelo_factory", "feedforward", layer_sizes=[2, 1]
        )
        self.assertIsInstance(model, Network)
        self.assertIs(model, self.alpha.get_model("modelo_factory"))

    def test_create_language_model(self):
        vocabulary = Vocabulary().build([["ola", "mundo"]])
        model = self.alpha.create_language_model(
            "modelo_linguagem", vocabulary, embedding_dim=4
        )
        self.assertIsInstance(model, LanguageModel)
        self.assertEqual(self.alpha.model_count(), 1)

    def test_unknown_model_type_does_not_register_model(self):
        with self.assertRaises(ValueError):
            self.alpha.create_model_by_type("modelo_invalido", "unknown")
        self.assertEqual(self.alpha.model_count(), 0)

    def test_run_model(self):
        self.alpha.create_model(
            "modelo_execucao", [1, 1], activation="linear"
        )
        self.assertEqual(len(self.alpha.run_model("modelo_execucao", [1.0])), 1)

    def test_generate_language_model(self):
        vocabulary = Vocabulary().build([["ola", "mundo"]])
        self.alpha.create_language_model(
            "modelo_geracao", vocabulary, embedding_dim=4,
            hidden_dim=4, max_length=8, seed=1
        )
        prompt = [vocabulary.get_id("ola")]
        generated = self.alpha.generate_tokens(
            "modelo_geracao", prompt, max_new_tokens=2
        )
        self.assertEqual(len(generated), 3)
        self.assertEqual(len(prompt), 1)

    def test_train_model(self):
        self.alpha.create_model(
            "modelo_treino",
            [1, 1],
            activation="linear"
        )

        dataset = Dataset([
            ([1.0], [2.0]),
            ([2.0], [4.0]),
        ])

        result = self.alpha.train_model(
            "modelo_treino",
            dataset,
            GradientDescent(learning_rate=0.1),
            epochs=5
        )

        self.assertEqual(result["epochs"], 5)
        self.assertEqual(len(result["history"]), 5)

    def test_evaluate_model(self):
        self.alpha.create_model(
            "modelo_avaliacao",
            [1, 1],
            activation="linear"
        )

        dataset = Dataset([
            ([1.0], [2.0]),
            ([2.0], [4.0]),
        ])

        result = self.alpha.evaluate(
            "modelo_avaliacao",
            dataset
        )

        self.assertEqual(result["samples"], 2)
        self.assertIn("metrics", result)

    def test_generate_report(self):
        self.alpha.create_model(
            "modelo_relatorio",
            [1, 1],
            activation="linear"
        )

        dataset = Dataset([
            ([1.0], [2.0]),
            ([2.0], [4.0]),
        ])

        report = self.alpha.report(
            "modelo_relatorio",
            dataset
        )

        self.assertIn("summary", report)
        self.assertIn("sample_reports", report)
        self.assertEqual(report["summary"]["samples"], 2)


if __name__ == "__main__":
    unittest.main()
