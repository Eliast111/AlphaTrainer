import json
import os
import tempfile
import unittest

from checkpoints import (
    save_model, load_model, save_language_model, load_language_model,
    load_any, save_any
)
from models.factory import create_model
from transformer.language_model import LanguageModel

CORPUS = ["o gato bebe leite .", "o cao come carne ."]


class TestLanguageCheckpoint(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "lm.json")
        self.model = LanguageModel.from_texts(
            CORPUS, embedding_dim=8, hidden_dim=8, max_length=10, seed=2
        )
        self.model.fit(CORPUS, epochs=5, learning_rate=0.02)

    def tearDown(self):
        self.dir.cleanup()

    def test_roundtrip_gives_identical_output(self):
        save_language_model(self.model, self.path)
        loaded = load_language_model(self.path)
        ids = self.model.encode_text("o gato")
        self.assertEqual(
            self.model.forward(ids).data, loaded.forward(ids).data
        )
        self.assertEqual(
            self.model.generate("o", temperature=0),
            loaded.generate("o", temperature=0)
        )

    def test_load_any_detects_type(self):
        net_path = os.path.join(self.dir.name, "net.json")
        save_any(create_model([2, 2, 1]), net_path)
        save_any(self.model, self.path)
        self.assertIsInstance(load_any(net_path), type(create_model([1, 1])))
        self.assertIsInstance(load_any(self.path), LanguageModel)

    def test_creates_missing_directories(self):
        nested = os.path.join(self.dir.name, "a", "b", "m.json")
        save_model(create_model([1, 1]), nested)
        self.assertTrue(os.path.isfile(nested))
        load_model(nested)

    def test_no_temp_files_left_behind(self):
        save_language_model(self.model, self.path)
        self.assertEqual(os.listdir(self.dir.name), ["lm.json"])

    def test_invalid_files(self):
        with open(self.path, "w") as file:
            json.dump({"format": "x"}, file)
        with self.assertRaises(ValueError):
            load_language_model(self.path)
        with self.assertRaises(ValueError):
            load_any(self.path)

    def test_corrupted_weights_are_rejected(self):
        save_language_model(self.model, self.path)
        with open(self.path) as file:
            data = json.load(file)
        data["weights"]["out_b"] = data["weights"]["out_b"][:-1]
        with open(self.path, "w") as file:
            json.dump(data, file)
        with self.assertRaises(ValueError):
            load_language_model(self.path)

    def test_missing_parameter_is_rejected(self):
        save_language_model(self.model, self.path)
        with open(self.path) as file:
            data = json.load(file)
        del data["weights"]["embedding"]
        with open(self.path, "w") as file:
            json.dump(data, file)
        with self.assertRaises(ValueError):
            load_language_model(self.path)

    def test_wrong_type_is_rejected(self):
        with self.assertRaises(TypeError):
            save_language_model(create_model([1, 1]), self.path)


if __name__ == "__main__":
    unittest.main()
