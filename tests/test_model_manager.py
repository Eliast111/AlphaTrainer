import unittest

from core import AlphaCore
from models import ModelFactory, ModelManager, PluginRegistry
from neural.network import Network


class TestModelManager(unittest.TestCase):
    def setUp(self):
        self.alpha = AlphaCore()

    def test_metadata_for_dense_model(self):
        self.alpha.create_model("dense", [2, 3, 1])
        metadata = self.alpha.model_metadata("dense")
        self.assertEqual(metadata["model_type"], "numeric")
        self.assertEqual(metadata["architecture"], "dense")
        self.assertEqual(metadata["config"]["layers"], [2, 3, 1])
        self.assertEqual(metadata["training_status"], "created")
        self.assertTrue(metadata["model_id"])

    def test_capabilities_are_modality_specific(self):
        self.alpha.create_model("dense", [1, 1])
        dense = self.alpha.model_capabilities("dense")
        self.assertTrue(dense["predict"])
        self.assertFalse(dense["generate"])
        self.alpha.create_language_model(
            "lm", ["ola mundo"], embedding_dim=4, hidden_dim=4
        )
        language = self.alpha.model_capabilities("lm")
        self.assertTrue(language["generate"])
        self.assertFalse(language["resume"])

    def test_language_model_metadata(self):
        self.alpha.create_language_model(
            "text", ["ola mundo"], embedding_dim=4, hidden_dim=4
        )
        metadata = self.alpha.model_metadata("text")
        self.assertEqual(metadata["model_type"], "text")
        self.assertEqual(metadata["architecture"], "transformer_decoder")

    def test_legacy_custom_object_is_preserved(self):
        value = object()
        self.alpha.register_model("external", value)
        self.assertIs(self.alpha.get_model("external"), value)
        self.assertEqual(self.alpha.model_metadata("external")["model_type"], "custom")
        self.assertFalse(any(self.alpha.model_capabilities("external").values()))

    def test_rename_and_duplicate(self):
        original = self.alpha.create_model("one", [1, 1])
        self.alpha.rename_model("one", "renamed")
        self.assertIs(self.alpha.get_model("renamed"), original)
        duplicate = self.alpha.duplicate_model("renamed", "copy")
        self.assertIsNot(duplicate, original)
        self.assertEqual(self.alpha.model_count(), 2)
        self.assertEqual(self.alpha.model_metadata("copy")["architecture"], "dense")

    def test_training_status_and_metrics(self):
        from data.samples import load_sample
        from learning.optimizer import GradientDescent
        self.alpha.create_model("train", [1, 1], activation="linear")
        result = self.alpha.train_model(
            "train", load_sample("linear"), GradientDescent(0.01), epochs=2
        )
        metadata = self.alpha.model_metadata("train")
        self.assertEqual(metadata["training_status"], "completed")
        self.assertEqual(metadata["metrics"][-1]["epochs_run"], result["epochs_run"])

    def test_factory_lists_supported_architectures(self):
        factory = ModelFactory()
        self.assertIn("dense", factory.architectures())
        self.assertIn("transformer_lm", factory.architectures())
        model = factory.create("dense", layer_sizes=[2, 1])
        self.assertIsInstance(model, Network)

    def test_factory_builds_existing_transformer_engine(self):
        from transformer.language_model import LanguageModel
        factory = ModelFactory()
        model = factory.create(
            "transformer_lm", texts=["ola mundo"],
            embedding_dim=4, hidden_dim=4
        )
        self.assertIsInstance(model, LanguageModel)

    def test_plugin_registry_requires_callable(self):
        plugins = PluginRegistry()
        with self.assertRaises(TypeError):
            plugins.register("bad", None)
        plugins.register("simple", lambda width: {"width": width})
        self.assertEqual(plugins.create("simple", width=4), {"width": 4})


    def test_metadata_can_be_exported_and_restored(self):
        self.alpha.create_model("persisted", [2, 3, 1])
        self.alpha.model_manager.update_training(
            "persisted", status="completed",
            metrics=[{"loss": 0.2, "epochs_run": 5}]
        )
        exported = self.alpha.model_manager.export_metadata("persisted")

        restored_alpha = AlphaCore()
        restored_alpha.create_model("persisted", [2, 3, 1])
        restored = restored_alpha.model_manager.restore_metadata(
            "persisted", exported
        )

        self.assertEqual(restored.model_id, exported["model_id"])
        self.assertEqual(restored.training_status, "completed")
        self.assertEqual(restored.metrics, exported["metrics"])

    def test_restore_metadata_rejects_architecture_mismatch(self):
        self.alpha.create_model("persisted", [2, 3, 1])
        exported = self.alpha.model_manager.export_metadata("persisted")
        exported["architecture"] = "unknown"

        with self.assertRaises(ValueError):
            self.alpha.model_manager.restore_metadata("persisted", exported)

    def test_restore_metadata_rejects_capability_mismatch(self):
        self.alpha.create_model("persisted", [2, 3, 1])
        exported = self.alpha.model_manager.export_metadata("persisted")
        exported["capabilities"]["generate"] = True

        with self.assertRaises(ValueError):
            self.alpha.model_manager.restore_metadata("persisted", exported)


if __name__ == "__main__":
    unittest.main()
