import unittest

from language.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_tokenize_words(self):
        result = self.tokenizer.tokenize("Alpha aprende linguagem")
        self.assertEqual(result, ["alpha", "aprende", "linguagem"])

    def test_lowercase(self):
        result = self.tokenizer.tokenize("ALPHA")
        self.assertEqual(result, ["alpha"])

    def test_punctuation(self):
        result = self.tokenizer.tokenize("Olá, mundo!")
        self.assertEqual(result, ["olá", ",", "mundo", "!"])

    def test_empty_text(self):
        self.assertEqual(self.tokenizer.tokenize(""), [])

    def test_numbers(self):
        result = self.tokenizer.tokenize("Alpha 2026")
        self.assertEqual(result, ["alpha", "2026"])

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            self.tokenizer.tokenize(123)

    def test_detokenize(self):
        result = self.tokenizer.detokenize(["alpha", "aprende"])
        self.assertEqual(result, "alpha aprende")

    def test_invalid_tokens(self):
        with self.assertRaises(TypeError):
            self.tokenizer.detokenize("alpha")

    def test_invalid_token_element(self):
        with self.assertRaises(TypeError):
            self.tokenizer.detokenize(["alpha", 123])


if __name__ == "__main__":
    unittest.main()
