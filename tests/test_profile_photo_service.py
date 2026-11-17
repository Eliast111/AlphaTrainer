import unittest

from cloud.profile_photo_service import (
    ProfilePhotoService,
    ProfilePhotoServiceError,
)


class TestProfilePhotoService(unittest.TestCase):

    def setUp(self):
        self.service = ProfilePhotoService.__new__(
            ProfilePhotoService
        )

    def test_accepts_jpeg(self):
        self.assertEqual(
            self.service._detect_extension(
                b"\xff\xd8\xff" + b"\x00" * 20
            ),
            "jpg",
        )

    def test_accepts_png(self):
        self.assertEqual(
            self.service._detect_extension(
                b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
            ),
            "png",
        )

    def test_accepts_webp(self):
        data = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 20
        self.assertEqual(
            self.service._detect_extension(data),
            "webp",
        )

    def test_rejects_unknown_format(self):
        with self.assertRaises(ProfilePhotoServiceError):
            self.service._detect_extension(b"imagem invalida")

    def test_rejects_empty_image(self):
        with self.assertRaises(ProfilePhotoServiceError):
            self.service._detect_extension(b"")

    def test_rejects_image_over_5mb(self):
        data = b"\xff\xd8\xff" + b"\x00" * self.service.MAX_SIZE
        with self.assertRaises(ProfilePhotoServiceError):
            self.service._detect_extension(data)


if __name__ == "__main__":
    unittest.main()
