import unittest

from language.vocabulary import Vocabulary


class TestVocabulary(unittest.TestCase):

    def setUp(self):
        self.vocab = Vocabulary()

    def test_special_tokens(self):
        self.assertEqual(self.vocab.pad_id, 0)
        self.assertEqual(self.vocab.unk_id, 1)
        self.assertEqual(self.vocab.bos_id, 2)
        self.assertEqual(self.vocab.eos_id, 3)

    def test_initial_size(self):
        self.assertEqual(self.vocab.size, 4)

    def test_add_token(self):
        token_id = self.vocab.add_token("alpha")
        self.assertEqual(token_id, 4)

    def test_duplicate_token(self):
        first = self.vocab.add_token("alpha")
        second = self.vocab.add_token("alpha")
        self.assertEqual(first, second)
        self.assertEqual(self.vocab.size, 5)

    def test_build_vocabulary(self):
        self.vocab.build([
            ["alpha", "aprende"],
            ["alpha", "linguagem"]
        ])
        self.assertEqual(self.vocab.size, 7)

    def test_get_unknown_token(self):
        self.assertEqual(self.vocab.get_id("desconhecido"), self.vocab.unk_id)

    def test_get_token(self):
        self.vocab.add_token("alpha")
        self.assertEqual(self.vocab.get_token(4), "alpha")

    def test_encode(self):
        self.vocab.build([["alpha", "aprende"]])
        self.assertEqual(self.vocab.encode(["alpha", "aprende"]), [4, 5])

    def test_encode_with_boundaries(self):
        self.vocab.add_token("alpha")
        result = self.vocab.encode(
            ["alpha"], add_bos=True, add_eos=True
        )
        self.assertEqual(result, [2, 4, 3])

    def test_decode(self):
        self.vocab.build([["alpha", "aprende"]])
        self.assertEqual(
            self.vocab.decode([4, 5]),
            ["alpha", "aprende"]
        )

    def test_decode_skip_special_tokens(self):
        self.vocab.add_token("alpha")
        result = self.vocab.decode([2, 4, 3], skip_special_tokens=True)
        self.assertEqual(result, ["alpha"])

    def test_invalid_token(self):
        with self.assertRaises(ValueError):
            self.vocab.add_token("")

    def test_invalid_build(self):
        with self.assertRaises(TypeError):
            self.vocab.build("alpha")

    def test_invalid_id(self):
        with self.assertRaises(ValueError):
            self.vocab.get_token(999)

    def test_invalid_id_type(self):
        with self.assertRaises(TypeError):
            self.vocab.get_token("4")


if __name__ == "__main__":
    unittest.main()
