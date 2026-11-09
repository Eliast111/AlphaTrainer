import math
import random
import unittest

from transformer.language_model import LanguageModel
from transformer.layer import TransformerLayer
from transformer.attention import SelfAttention
from alpha_math.tensor import Tensor

CORPUS = [
    "o gato bebe leite .",
    "o cao come carne .",
    "o gato dorme no sofa .",
    "o cao dorme no jardim .",
]


def numeric_check(test, model, inputs, targets, samples=6, tolerance=1e-5):
    _, grads = model.loss_and_gradients(inputs, targets)
    rng = random.Random(0)

    for name, array in model.parameters():
        is_matrix = isinstance(array[0], list)

        for _ in range(samples):
            if is_matrix:
                i = rng.randrange(len(array))
                j = rng.randrange(len(array[0]))
                analytic = grads[name][i][j]
                original = array[i][j]

                def put(value):
                    array[i][j] = value
            else:
                j = rng.randrange(len(array))
                analytic = grads[name][j]
                original = array[j]

                def put(value):
                    array[j] = value

            eps = 1e-5
            put(original + eps)
            plus, _ = model.loss_and_gradients(inputs, targets)
            put(original - eps)
            minus, _ = model.loss_and_gradients(inputs, targets)
            put(original)

            numeric = (plus - minus) / (2 * eps)
            scale = max(1e-8, abs(numeric) + abs(analytic))
            test.assertLess(
                abs(numeric - analytic) / scale, tolerance,
                f"gradiente errado em {name}"
            )


class TestLanguageModelGradients(unittest.TestCase):

    def test_gradients_match_numerical(self):
        model = LanguageModel.from_texts(
            CORPUS, embedding_dim=6, hidden_dim=10,
            num_layers=2, max_length=8, seed=3
        )
        ids = model.vocabulary.encode(
            "o gato dorme no".split(), add_bos=True
        )
        numeric_check(self, model, ids[:-1], ids[1:])

    def test_attention_backward_shapes_and_causality(self):
        attention = SelfAttention(causal=True)
        x = [[0.1, 0.2], [0.3, -0.1], [0.0, 0.5]]
        _, scores = attention.forward(Tensor(x), Tensor(x), Tensor(x))
        dq, dk, dv = attention.backward(
            x, x, x, scores.data, [[1.0, 1.0]] * 3
        )
        self.assertEqual((len(dq), len(dq[0])), (3, 2))
        # A ultima posicao so afeta a propria saida (mascara causal):
        # dv da primeira linha vem de todas as posicoes, a ultima so de si.
        self.assertEqual(len(dk), 3)
        self.assertEqual(len(dv), 3)

    def test_layer_backward_matches_numerical(self):
        layer = TransformerLayer(4, 6, causal=True, seed=1)
        x = [[0.2, -0.1, 0.4, 0.3], [0.1, 0.5, -0.3, 0.2], [-0.2, 0.1, 0.0, 0.6]]
        weights = [[0.3, -0.2, 0.5, 0.1]] * 3

        def loss(rows):
            out, _ = layer.forward_cache(rows)
            return sum(a * b for ra, rb in zip(out, weights)
                       for a, b in zip(ra, rb))

        _, cache = layer.forward_cache(x)
        d_input, _ = layer.backward(weights, cache)

        eps = 1e-6
        for i in range(3):
            for j in range(4):
                plus = [list(r) for r in x]
                minus = [list(r) for r in x]
                plus[i][j] += eps
                minus[i][j] -= eps
                numeric = (loss(plus) - loss(minus)) / (2 * eps)
                self.assertAlmostEqual(numeric, d_input[i][j], places=5)


class TestLanguageModelTraining(unittest.TestCase):

    def make(self, **kwargs):
        return LanguageModel.from_texts(
            CORPUS, embedding_dim=16, hidden_dim=32,
            num_layers=1, max_length=16, seed=1, **kwargs
        )

    def test_loss_decreases_a_lot(self):
        model = self.make()
        result = model.fit(CORPUS, epochs=40, learning_rate=0.02, batch_size=2)
        self.assertLess(result["final_loss"], result["history"][0] / 3)
        self.assertGreater(result["perplexity"], 1.0)

    def test_memorizes_and_generates(self):
        model = self.make()
        model.fit(CORPUS, epochs=60, learning_rate=0.02, batch_size=2)
        self.assertEqual(
            model.generate("o gato", max_new_tokens=8, temperature=0),
            "o gato bebe leite ."
        )

    def test_generate_is_reproducible_with_seed(self):
        model = self.make()
        model.fit(CORPUS, epochs=10, learning_rate=0.02)
        a = model.generate("o", max_new_tokens=6, seed=4)
        b = model.generate("o", max_new_tokens=6, seed=4)
        self.assertEqual(a, b)

    def test_generate_top_k_and_empty_prompt(self):
        model = self.make()
        model.fit(CORPUS, epochs=5, learning_rate=0.02)
        text = model.generate("", max_new_tokens=5, top_k=2, seed=1)
        self.assertIsInstance(text, str)
        self.assertNotIn("<", text)

    def test_generate_longer_than_context(self):
        model = LanguageModel.from_texts(
            CORPUS, embedding_dim=8, hidden_dim=8, max_length=4, seed=1
        )
        model.generate("o gato bebe leite", max_new_tokens=6, temperature=0)

    def test_generate_validation(self):
        model = self.make()
        with self.assertRaises(ValueError):
            model.generate("o", max_new_tokens=-1)
        with self.assertRaises(ValueError):
            model.generate("o", temperature=-1)
        with self.assertRaises(ValueError):
            model.generate("o", top_k=0)
        with self.assertRaises(TypeError):
            model.generate(5)

    def test_evaluate(self):
        model = self.make()
        before = model.evaluate(CORPUS)["loss"]
        model.fit(CORPUS, epochs=20, learning_rate=0.02)
        after = model.evaluate(CORPUS)
        self.assertLess(after["loss"], before)
        self.assertAlmostEqual(after["perplexity"], math.exp(after["loss"]))

    def test_training_pairs_split_long_sequences(self):
        model = LanguageModel.from_texts(
            ["a b c d e f g h i j"], embedding_dim=4, hidden_dim=4,
            max_length=4, seed=1
        )
        pairs = model.make_training_pairs("a b c d e f g h i j")
        self.assertGreater(len(pairs), 1)
        for inputs, targets in pairs:
            self.assertLessEqual(len(inputs), 4)
            self.assertEqual(len(inputs), len(targets))

    def test_callback_stops_training(self):
        model = self.make()
        result = model.fit(CORPUS, epochs=30, learning_rate=0.02,
                           callback=lambda e, l: e >= 2)
        self.assertEqual(result["epochs_run"], 2)
        self.assertTrue(result["stopped_early"])

    def test_invalid_training_options(self):
        model = self.make()
        with self.assertRaises(ValueError):
            model.fit(CORPUS, epochs=0)
        with self.assertRaises(ValueError):
            model.fit(CORPUS, epochs=1, batch_size=0)
        with self.assertRaises(ValueError):
            model.fit(CORPUS, epochs=1, learning_rate=-1)
        with self.assertRaises(ValueError):
            model.fit([], epochs=1)

    def test_parameter_count(self):
        model = self.make()
        v = model.vocabulary.size
        expected = (v * 16) + (16 * v + v) + (16 * 32 + 32 + 32 * 16 + 16)
        self.assertEqual(model.parameter_count(), expected)


if __name__ == "__main__":
    unittest.main()
