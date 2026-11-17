import os
import unittest
from unittest.mock import patch

from config.settings import Settings, DEFAULT_FIREBASE_DATABASE_URL


class TestSettings(unittest.TestCase):

    def test_default_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()

        self.assertEqual(settings.firebase_api_key, "")
        self.assertEqual(
            settings.firebase_database_url,
            DEFAULT_FIREBASE_DATABASE_URL,
        )
        self.assertEqual(settings.supabase_url, "")
        self.assertEqual(settings.supabase_publishable_key, "")

    def test_reads_environment_variables(self):
        values = {
            "ALPHATRAINER_FIREBASE_API_KEY": "firebase-test",
            "ALPHATRAINER_FIREBASE_DATABASE_URL":
                "https://firebase.example.com/",
            "ALPHATRAINER_SUPABASE_URL":
                "https://supabase.example.com/",
            "ALPHATRAINER_SUPABASE_PUBLISHABLE_KEY":
                "supabase-test",
        }

        with patch.dict(os.environ, values, clear=True):
            settings = Settings.from_env()

        self.assertEqual(settings.firebase_api_key, "firebase-test")
        self.assertEqual(
            settings.firebase_database_url,
            "https://firebase.example.com",
        )
        self.assertEqual(
            settings.supabase_url,
            "https://supabase.example.com",
        )
        self.assertEqual(
            settings.supabase_publishable_key,
            "supabase-test",
        )

    def test_supports_legacy_supabase_environment_names(self):
        values = {
            "SUPABASE_URL": "https://legacy.example.com",
            "SUPABASE_PUBLISHABLE_KEY": "legacy-test",
        }

        with patch.dict(os.environ, values, clear=True):
            settings = Settings.from_env()

        self.assertEqual(
            settings.supabase_url,
            "https://legacy.example.com",
        )
        self.assertEqual(
            settings.supabase_publishable_key,
            "legacy-test",
        )

    def test_settings_are_immutable(self):
        settings = Settings()

        with self.assertRaises(AttributeError):
            settings.firebase_api_key = "changed"


if __name__ == "__main__":
    unittest.main()
