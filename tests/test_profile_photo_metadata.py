import unittest
from unittest.mock import Mock, patch

from cloud.profile_photo_metadata import (
    ProfilePhotoMetadata,
    ProfilePhotoMetadataError,
)


class TestProfilePhotoMetadata(unittest.TestCase):
    def setUp(self):
        self.session = {
            "uid": "uid_teste",
            "id_token": "token_teste",
        }
        self.metadata = ProfilePhotoMetadata(self.session)

    def test_profile_path_isolated_by_uid(self):
        self.assertEqual(
            self.metadata._profile_path(),
            "users/uid_teste/profile",
        )

    def test_missing_uid_is_rejected(self):
        metadata = ProfilePhotoMetadata({"id_token": "token"})
        with self.assertRaises(ProfilePhotoMetadataError):
            metadata.get_photo_path()

    def test_save_photo_path(self):
        with patch.object(self.metadata.sync, "_request") as request:
            result = self.metadata.set_photo_path("uid_teste/avatar.jpg")

        self.assertEqual(result, "uid_teste/avatar.jpg")
        request.assert_called_once()
        method, path, record = request.call_args.args
        self.assertEqual(method, "PATCH")
        self.assertEqual(path, "users/uid_teste/profile")
        self.assertEqual(record["photo_path"], "uid_teste/avatar.jpg")
        self.assertIn("updated_at", record)

    def test_get_photo_path(self):
        with patch.object(
            self.metadata.sync,
            "_request",
            return_value={"photo_path": "uid_teste/avatar.webp"},
        ):
            result = self.metadata.get_photo_path()

        self.assertEqual(result, "uid_teste/avatar.webp")

    def test_get_missing_photo_path(self):
        with patch.object(
            self.metadata.sync, "_request", return_value=None
        ):
            self.assertIsNone(self.metadata.get_photo_path())

    def test_reject_photo_from_another_user(self):
        with self.assertRaises(ProfilePhotoMetadataError):
            self.metadata.set_photo_path("outro_uid/avatar.jpg")

    def test_reject_invalid_filename(self):
        with self.assertRaises(ProfilePhotoMetadataError):
            self.metadata.set_photo_path("uid_teste/foto.jpg")

    def test_clear_photo_path(self):
        with patch.object(self.metadata.sync, "_request") as request:
            result = self.metadata.clear_photo_path()

        self.assertTrue(result)
        method, path, record = request.call_args.args
        self.assertEqual(method, "PATCH")
        self.assertIsNone(record["photo_path"])
        self.assertIn("updated_at", record)


if __name__ == "__main__":
    unittest.main()
