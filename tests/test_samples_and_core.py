import os
import tempfile
import unittest

from core import AlphaCore
from data.samples import available_samples, load_sample, SAMPLES
from learning.optimizer import Adam
from alpha_math.linalg import matmul, transpose, add, column_sums


class TestSamples(unittest.TestCase):

    def test_all_samples_load(self):
        for name in SAMPLES:
            dataset = load_sample(name)
            self.assertGreater(len(dataset), 0)
        self.assertEqual(set(available_samples()), set(SAMPLES))

    def test_xor_truth_table(self):
        data = load_sample("XOR")
        self.assertEqual(
            sorted((tuple(i), t[0]) for i, t in data),
            [((0.0, 0.0), 0.0), ((0.0, 1.0), 1.0),
             ((1.0, 0.0), 1.0), ((1.0, 1.0), 0.0)]
        )

    def test_unknown_sample(self):
        with self.assertRaises(ValueError):
            load_sample("nada")


class TestLinalg(unittest.TestCase):

    def test_operations(self):
        a = [[1, 2], [3, 4]]
        self.assertEqual(matmul(a, [[1, 0], [0, 1]]), a)
        self.assertEqual(transpose(a), [[1, 3], [2, 4]])
        self.assertEqual(add(a, a), [[2, 4], [6, 8]])
        self.assertEqual(column_sums(a), [4, 6])


class TestCoreExtended(unittest.TestCase):

    def setUp(self):
        self.alpha = AlphaCore()

    def test_duplicate_name_rejected(self):
        self.alpha.create_model("m", [1, 1])
        with self.assertRaises(ValueError):
            self.alpha.create_model("m", [1, 1])

    def test_describe_network(self):
        self.alpha.create_model("m", [2, 3, 1], activation="tanh",
                                output_activation="sigmoid")
        info = self.alpha.describe_model("m")
        self.assertEqual(info["layers"], [2, 3, 1])
        self.assertEqual(info["activations"], ["tanh", "sigmoid"])

    def test_train_predict_save_load(self):
        self.alpha.create_model("m", [1, 1], activation="linear")
        self.alpha.train_model("m", load_sample("linear"), Adam(0.1),
                               epochs=200)
        value = self.alpha.predict("m", [0.5])[0]
        self.assertAlmostEqual(value, 2.0, places=1)

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "m.json")
            self.alpha.save_model("m", path)
            other = AlphaCore()
            other.load_model("copia", path)
            self.assertEqual(other.predict("copia", [0.5]),
                             self.alpha.predict("m", [0.5]))
            with self.assertRaises(ValueError):
                other.load_model("copia", path)

    def test_language_model_workflow(self):
        texts = ["o gato dorme .", "o cao come ."]
        self.alpha.create_language_model(
            "lm", texts, embedding_dim=8, hidden_dim=8, max_length=8
        )
        result = self.alpha.train_language_model(
            "lm", texts, epochs=30, learning_rate=0.03
        )
        self.assertLess(result["final_loss"], result["history"][0])
        self.assertIsInstance(self.alpha.generate("lm", "o"), str)
        self.assertEqual(self.alpha.describe_model("lm")["type"],
                         "modelo de linguagem")

    def test_type_mismatches(self):
        self.alpha.create_model("m", [1, 1])
        self.alpha.create_language_model("lm", ["a b c"],
                                         embedding_dim=4, hidden_dim=4)
        with self.assertRaises(TypeError):
            self.alpha.generate("m", "a")
        with self.assertRaises(TypeError):
            self.alpha.predict("lm", [1.0])
        with self.assertRaises(TypeError):
            self.alpha.train_language_model("m", ["a b"])

    def test_missing_model(self):
        with self.assertRaises(KeyError):
            self.alpha.predict("nada", [1.0])


if __name__ == "__main__":
    unittest.main()
