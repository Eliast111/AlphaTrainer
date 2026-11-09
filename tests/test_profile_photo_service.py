import pytest

from cloud.profile_photo_service import (
    ProfilePhotoService,
    ProfilePhotoServiceError,
)


@pytest.fixture
def service():
    return ProfilePhotoService.__new__(ProfilePhotoService)


def test_accepts_jpeg(service):
    assert service._detect_extension(
        b"\xff\xd8\xff" + b"\x00" * 20
    ) == "jpg"


def test_accepts_png(service):
    assert service._detect_extension(
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    ) == "png"


def test_accepts_webp(service):
    data = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 20
    assert service._detect_extension(data) == "webp"


def test_rejects_unknown_format(service):
    with pytest.raises(ProfilePhotoServiceError):
        service._detect_extension(b"imagem invalida")


def test_rejects_empty_image(service):
    with pytest.raises(ProfilePhotoServiceError):
        service._detect_extension(b"")


def test_rejects_image_over_5mb(service):
    data = b"\xff\xd8\xff" + b"\x00" * (
        service.MAX_SIZE
    )
    with pytest.raises(ProfilePhotoServiceError):
        service._detect_extension(data)
