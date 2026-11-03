import json
import os
import tempfile
import unittest

from checkpoints.checkpoint import save_model, load_model
from models.factory import create_model


class TestAlphaCheckpoint(unittest.TestCase):

    def setUp(self):
        self.model = create_model([2, 3, 1])

    def test_save_model(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.json")
            save_model(self.model, path)
            self.assertTrue(os.path.exists(path))

    def test_load_model(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.json")
            save_model(self.model, path)
            loaded = load_model(path)
            self.assertEqual(loaded.layer_count(), 2)

    def test_predictions_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.json")
            expected = self.model.forward([1.0, 2.0])
            save_model(self.model, path)
            loaded = load_model(path)
            self.assertEqual(loaded.forward([1.0, 2.0]), expected)

    def test_weights_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.json")
            save_model(self.model, path)
            loaded = load_model(path)

            for old_layer, new_layer in zip(
                self.model.layers, loaded.layers
            ):
                for old, new in zip(
                    old_layer.neurons, new_layer.neurons
                ):
                    self.assertEqual(old.weights, new.weights)

    def test_invalid_model_type(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.json")
            with self.assertRaises(TypeError):
                save_model(None, path)

    def test_invalid_format(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.json")

            with open(path, "w") as file:
                json.dump({"format": "wrong"}, file)

            with self.assertRaises(ValueError):
                load_model(path)

    def test_invalid_json_structure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.json")

            with open(path, "w") as file:
                json.dump([], file)

            with self.assertRaises(ValueError):
                load_model(path)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_model("arquivo_inexistente_alpha.json")


if __name__ == "__main__":
    unittest.main()
