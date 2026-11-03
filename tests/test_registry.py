import unittest

from registry.registry import ModelRegistry
from models.factory import create_model


class TestAlphaRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = ModelRegistry()
        self.model = create_model([2, 3, 1])

    def test_register_model(self):
        self.registry.register("alpha", self.model)
        self.assertEqual(self.registry.count(), 1)

    def test_get_model(self):
        self.registry.register("alpha", self.model)
        self.assertIs(self.registry.get("alpha"), self.model)

    def test_list_models(self):
        self.registry.register("alpha", self.model)
        self.assertEqual(self.registry.list_models(), ["alpha"])

    def test_remove_model(self):
        self.registry.register("alpha", self.model)
        removed = self.registry.remove("alpha")
        self.assertIs(removed, self.model)
        self.assertEqual(self.registry.count(), 0)

    def test_duplicate_name(self):
        self.registry.register("alpha", self.model)
        with self.assertRaises(ValueError):
            self.registry.register("alpha", self.model)

    def test_invalid_name(self):
        with self.assertRaises(ValueError):
            self.registry.register("", self.model)

    def test_missing_model(self):
        with self.assertRaises(KeyError):
            self.registry.get("inexistente")

    def test_remove_missing_model(self):
        with self.assertRaises(KeyError):
            self.registry.remove("inexistente")


if __name__ == "__main__":
    unittest.main()
