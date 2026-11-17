import unittest
from language.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_words_and_punctuation(self):
        self.assertEqual(
            self.tokenizer.tokenize("Olá, mundo! Alpha 2.0."),
            ["Olá", ",", "mundo", "!", "Alpha", "2", ".", "0", "."]
        )

    def test_unicode_and_apostrophe(self):
        self.assertEqual(
            self.tokenizer.tokenize("não d’água"),
            ["não", "d’água"]
        )

    def test_detokenize(self):
        self.assertEqual(
            self.tokenizer.detokenize(["Olá", ",", "mundo", "!"]),
            "Olá , mundo !"
        )

    def test_invalid_text(self):
        with self.assertRaises(TypeError):
            self.tokenizer.tokenize(123)

    def test_invalid_tokens(self):
        with self.assertRaises(TypeError):
            self.tokenizer.detokenize("texto")
        with self.assertRaises(TypeError):
            self.tokenizer.detokenize(["ok", 1])


if __name__ == "__main__":
    unittest.main()
