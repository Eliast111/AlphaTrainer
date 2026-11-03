import unittest

from language.vocabulary import Vocabulary
from transformer.language_model import LanguageModel
from alpha_math.tensor import Tensor


class TestLanguageModel(unittest.TestCase):

    def setUp(self):
        self.vocabulary = Vocabulary()
        self.vocabulary.build([
            ["ola", "mundo"],
            ["alpha", "inteligencia", "artificial"]
        ])

        self.model = LanguageModel(
            vocabulary=self.vocabulary,
            embedding_dim=8,
            hidden_dim=16,
            num_layers=2,
            max_length=20
        )

        self.ids = self.vocabulary.encode(["ola", "mundo"])

    def test_creation(self):
        self.assertEqual(self.model.embedding_dim, 8)
        self.assertEqual(self.model.hidden_dim, 16)
        self.assertEqual(self.model.num_layers, 2)

    def test_forward_returns_tensor(self):
        result = self.model.forward(self.ids)
        self.assertIsInstance(result, Tensor)

    def test_forward_shape(self):
        result = self.model.forward(self.ids)
        self.assertEqual(
            result.shape(),
            (len(self.ids), self.vocabulary.size)
        )

    def test_probabilities_sum_to_one(self):
        result = self.model.forward(self.ids)

        for row in result.data:
            self.assertAlmostEqual(sum(row), 1.0)

    def test_probabilities_are_positive(self):
        result = self.model.forward(self.ids)

        for row in result.data:
            for value in row:
                self.assertGreater(value, 0.0)

    def test_predict_next(self):
        prediction = self.model.predict_next(self.ids)

        self.assertIsInstance(prediction, int)
        self.assertGreaterEqual(prediction, 0)
        self.assertLess(prediction, self.vocabulary.size)

    def test_encode_text(self):
        result = self.model.encode_text("ola mundo")

        self.assertEqual(result, self.ids)

    def test_predict_next_text(self):
        result = self.model.predict_next_text("ola mundo")

        self.assertIsInstance(result, int)

    def test_decode(self):
        result = self.model.decode(self.ids)

        self.assertEqual(result, "ola mundo")

    def test_single_token(self):
        result = self.model.forward([self.vocabulary.get_id("ola")])

        self.assertEqual(result.shape(), (1, self.vocabulary.size))

    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            self.model.forward("ola")

    def test_empty_sequence(self):
        with self.assertRaises(ValueError):
            self.model.forward([])

    def test_invalid_token_id(self):
        with self.assertRaises(ValueError):
            self.model.forward([999])

    def test_invalid_token_type(self):
        with self.assertRaises(TypeError):
            self.model.forward(["ola"])

    def test_sequence_too_long(self):
        ids = [self.vocabulary.get_id("ola")] * 21

        with self.assertRaises(ValueError):
            self.model.forward(ids)

    def test_invalid_vocabulary(self):
        with self.assertRaises(TypeError):
            LanguageModel("vocabulary")

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            LanguageModel(self.vocabulary, embedding_dim=0)

    def test_invalid_num_layers(self):
        with self.assertRaises(ValueError):
            LanguageModel(self.vocabulary, num_layers=0)


if __name__ == "__main__":
    unittest.main()
