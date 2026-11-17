import json
import os
import tempfile
import unittest
from unittest.mock import patch

from checkpoints import save_model
from core import AlphaCore
from data.dataset import Dataset
from persistence import PersistenceManager
from transformer.language_model import LanguageModel


class TestVersionedPersistence(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.manager = PersistenceManager(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dense_metadata_survives_restart(self):
        alpha = AlphaCore()
        alpha.create_model("dense", [2, 3, 1])
        alpha.model_manager.update_training(
            "dense",
            status="completed",
            metrics=[{"loss": 0.125, "epochs_run": 7}],
            dataset_id="dataset-7",
        )
        before = alpha.model_metadata("dense")
        expected = alpha.predict("dense", [0.25, 0.75])

        self.manager.save_all(alpha, {})

        restored = AlphaCore()
        self.manager.load_all(restored)
        after = restored.model_metadata("dense")

        self.assertEqual(restored.predict("dense", [0.25, 0.75]), expected)
        self.assertEqual(after["model_id"], before["model_id"])
        self.assertEqual(after["training_status"], "completed")
        self.assertEqual(after["metrics"], before["metrics"])
        self.assertEqual(after["dataset_id"], "dataset-7")
        self.assertEqual(
            restored.model_capabilities("dense"),
            alpha.model_capabilities("dense"),
        )

    def test_transformer_survives_restart(self):
        alpha = AlphaCore()
        texts = ["o gato bebe leite .", "o cao come carne ."]
        alpha.create_language_model(
            "language", texts, embedding_dim=4, hidden_dim=4,
            max_length=8, seed=3
        )
        alpha.model_manager.update_training(
            "language", status="completed",
            metrics=[{"loss": 0.75}], dataset_id="text-corpus"
        )
        expected = alpha.generate("language", "o", temperature=0)
        before = alpha.model_metadata("language")

        self.manager.save_all(alpha, {})

        restored = AlphaCore()
        self.manager.load_all(restored)
        after = restored.model_metadata("language")

        self.assertEqual(
            restored.generate("language", "o", temperature=0),
            expected,
        )
        self.assertEqual(after["model_id"], before["model_id"])
        self.assertEqual(after["architecture"], "transformer_decoder")
        self.assertEqual(after["metrics"], before["metrics"])

    def test_v2_manifest_uses_versioned_files(self):
        alpha = AlphaCore()
        alpha.create_model("dense", [1, 1])
        self.manager.save_all(alpha, {})

        with open(self.manager.manifest_path, encoding="utf-8") as file:
            manifest = json.load(file)

        self.assertEqual(manifest["format"], "alpha_persistence_v2")
        self.assertEqual(manifest["format_version"], 2)
        filename = manifest["model_files"]["dense"]
        self.assertTrue(filename.endswith(".json"))
        self.assertTrue((self.manager.models_dir / filename).is_file())

    def test_legacy_v1_manifest_still_loads(self):
        alpha = AlphaCore()
        model = alpha.create_model("legacy", [2, 2, 1])
        path = self.manager.models_dir / self.manager._filename("legacy")
        save_model(model, path)

        self.manager.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self.manager.manifest_path, "w", encoding="utf-8") as file:
            json.dump({
                "format": "alpha_persistence_v1",
                "models": ["legacy"],
                "datasets": [],
            }, file)

        restored = AlphaCore()
        self.manager.load_all(restored)

        self.assertIn("legacy", restored.list_models())
        self.assertEqual(
            restored.model_metadata("legacy")["training_status"], "created"
        )

    def test_dataset_and_model_restore_from_v2(self):
        alpha = AlphaCore()
        alpha.create_model("dense", [2, 1])
        data = {"dense": Dataset([([0, 0], [0]), ([1, 1], [1])])}

        self.manager.save_all(alpha, data)
        restored = AlphaCore()
        datasets = self.manager.load_all(restored)

        self.assertIn("dense", datasets)
        self.assertEqual(len(datasets["dense"]), 2)

    def test_corrupt_model_does_not_block_other_models(self):
        alpha = AlphaCore()
        alpha.create_model("one", [1, 1])
        alpha.create_model("two", [1, 1])
        self.manager.save_all(alpha, {})

        with open(self.manager.manifest_path, encoding="utf-8") as file:
            manifest = json.load(file)
        bad_path = self.manager.models_dir / manifest["model_files"]["one"]
        bad_path.write_text("{broken", encoding="utf-8")

        restored = AlphaCore()
        self.manager.load_all(restored)

        self.assertNotIn("one", restored.list_models())
        self.assertIn("two", restored.list_models())
        self.assertTrue(self.manager.warnings)

    def test_unsafe_manifest_filename_is_rejected(self):
        alpha = AlphaCore()
        alpha.create_model("dense", [1, 1])
        self.manager.save_all(alpha, {})

        with open(self.manager.manifest_path, encoding="utf-8") as file:
            manifest = json.load(file)
        manifest["model_files"]["dense"] = "../outside.json"
        with open(self.manager.manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file)

        restored = AlphaCore()
        self.manager.load_all(restored)

        self.assertNotIn("dense", restored.list_models())
        self.assertTrue(self.manager.warnings)

    def test_failed_save_keeps_previous_manifest_and_files(self):
        alpha = AlphaCore()
        alpha.create_model("dense", [1, 1])
        self.manager.save_all(alpha, {})
        previous_manifest = self.manager.manifest_path.read_bytes()

        with patch("persistence.manager.save_any", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                self.manager.save_all(alpha, {})

        self.assertEqual(
            self.manager.manifest_path.read_bytes(), previous_manifest
        )
        restored = AlphaCore()
        self.manager.load_all(restored)
        self.assertIn("dense", restored.list_models())

    def test_custom_model_is_skipped_with_warning(self):
        alpha = AlphaCore()
        alpha.register_model("custom", object())

        self.manager.save_all(alpha, {})
        self.assertTrue(any(
            "sem checkpoint suportado" in w for w in self.manager.warnings
        ))

        restored = AlphaCore()
        self.manager.load_all(restored)
        self.assertNotIn("custom", restored.list_models())


    def test_duplicate_rename_and_remove_survive_save(self):
        alpha = AlphaCore()
        alpha.create_model("source", [1, 1])
        original_id = alpha.model_metadata("source")["model_id"]
        alpha.duplicate_model("source", "copy")
        alpha.rename_model("source", "renamed")
        alpha.remove_model("copy")

        self.manager.save_all(alpha, {})
        restored = AlphaCore()
        self.manager.load_all(restored)

        self.assertEqual(restored.list_models(), ["renamed"])
        self.assertEqual(
            restored.model_metadata("renamed")["model_id"], original_id
        )

    def test_invalid_metadata_does_not_discard_valid_model(self):
        alpha = AlphaCore()
        alpha.create_model("dense", [1, 1])
        self.manager.save_all(alpha, {})

        with open(self.manager.manifest_path, encoding="utf-8") as file:
            manifest = json.load(file)
        manifest["model_metadata"]["dense"]["architecture"] = "tampered"
        with open(self.manager.manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file)

        restored = AlphaCore()
        self.manager.load_all(restored)

        self.assertIn("dense", restored.list_models())
        self.assertTrue(any("Metadados de dense nao restaurados" in w
                            for w in self.manager.warnings))
        self.assertEqual(
            restored.model_metadata("dense")["architecture"], "dense"
        )

    def test_legacy_v1_dataset_still_loads(self):
        alpha = AlphaCore()
        model = alpha.create_model("legacy", [2, 1])
        self.manager.models_dir.mkdir(parents=True, exist_ok=True)
        self.manager.datasets_dir.mkdir(parents=True, exist_ok=True)
        save_model(
            model,
            self.manager.models_dir / self.manager._filename("legacy")
        )
        with open(
            self.manager.datasets_dir / self.manager._filename("legacy"),
            "w", encoding="utf-8"
        ) as file:
            json.dump({
                "format": "alpha_dataset_v1",
                "samples": [[[0, 0], [0]], [[1, 1], [1]]],
            }, file)
        with open(self.manager.manifest_path, "w", encoding="utf-8") as file:
            json.dump({
                "format": "alpha_persistence_v1",
                "models": ["legacy"],
                "datasets": ["legacy"],
            }, file)

        restored = AlphaCore()
        datasets = self.manager.load_all(restored)
        self.assertIn("legacy", datasets)
        self.assertEqual(len(datasets["legacy"]), 2)


if __name__ == "__main__":
    unittest.main()
