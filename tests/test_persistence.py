import json
import tempfile
import unittest
from pathlib import Path

from core import AlphaCore
from data.dataset import Dataset
from persistence import PersistenceManager


class TestPersistenceManager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.manager = PersistenceManager(self.temp_dir.name)

        self.alpha = AlphaCore()
        self.model = self.alpha.create_model(
            "Alpha_Test",
            [2, 3, 1]
        )

        self.dataset = Dataset([
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [2])
        ])

        self.datasets = {
            "Alpha_Test": self.dataset
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_all_creates_manifest(self):
        self.manager.save_all(self.alpha, self.datasets)

        self.assertTrue(self.manager.manifest_path.exists())

    def test_save_and_load_model(self):
        self.manager.save_all(self.alpha, self.datasets)

        restored = AlphaCore()
        self.manager.load_all(restored)

        self.assertIn("Alpha_Test", restored.list_models())

    def test_model_predictions_are_preserved(self):
        expected = self.model.forward([1.0, 2.0])

        self.manager.save_all(self.alpha, self.datasets)

        restored = AlphaCore()
        self.manager.load_all(restored)

        actual = restored.get_model("Alpha_Test").forward([1.0, 2.0])

        self.assertEqual(actual, expected)

    def test_dataset_is_restored(self):
        self.manager.save_all(self.alpha, self.datasets)

        restored = AlphaCore()
        datasets = self.manager.load_all(restored)

        self.assertIn("Alpha_Test", datasets)
        self.assertEqual(len(datasets["Alpha_Test"]), 4)

    def test_missing_manifest_returns_empty_datasets(self):
        restored = AlphaCore()

        datasets = self.manager.load_all(restored)

        self.assertEqual(datasets, {})

    def test_invalid_manifest_generates_warning(self):
        self.manager.base_dir.mkdir(parents=True, exist_ok=True)

        with open(self.manager.manifest_path, "w") as file:
            json.dump({"format": "wrong"}, file)

        restored = AlphaCore()
        datasets = self.manager.load_all(restored)

        self.assertEqual(datasets, {})
        self.assertTrue(self.manager.warnings)


if __name__ == "__main__":
    unittest.main()
