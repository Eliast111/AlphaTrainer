"""Metadados das fotografias de perfil no Firebase RTDB."""

import re
import time

from cloud.firebase_sync import FirebaseSync, FirebaseSyncError


class ProfilePhotoMetadataError(Exception):
    pass


class ProfilePhotoMetadata:
    ALLOWED_FILES = {"avatar.jpg", "avatar.jpeg", "avatar.png", "avatar.webp"}

    def __init__(
        self, session, database_url=None, timeout=20,
        session_manager=None
    ):
        kwargs = {
            "timeout": timeout,
            "session_manager": session_manager,
        }
        if database_url is not None:
            kwargs["database_url"] = database_url

        self.sync = FirebaseSync(session, **kwargs)

    def _uid(self):
        uid = self.sync.session.get("uid")
        if not isinstance(uid, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", uid
        ):
            raise ProfilePhotoMetadataError(
                "UID do utilizador inexistente ou invalido."
            )
        return uid

    def _profile_path(self):
        return f"users/{self._uid()}/profile"

    def _validate_photo_path(self, photo_path):
        uid = self._uid()

        if not isinstance(photo_path, str):
            raise ProfilePhotoMetadataError("Caminho da fotografia invalido.")

        parts = photo_path.split("/")
        if len(parts) != 2 or parts[0] != uid:
            raise ProfilePhotoMetadataError(
                "A fotografia nao pertence a este utilizador."
            )

        if parts[1] not in self.ALLOWED_FILES:
            raise ProfilePhotoMetadataError(
                "Formato ou nome de fotografia nao permitido."
            )

        return photo_path

    def get_photo_path(self):
        try:
            data = self.sync._request("GET", self._profile_path()) or {}
        except FirebaseSyncError:
            raise

        if not isinstance(data, dict):
            return None

        photo_path = data.get("photo_path")
        if photo_path is None:
            return None

        return self._validate_photo_path(photo_path)

    def set_photo_path(self, photo_path):
        photo_path = self._validate_photo_path(photo_path)

        record = {
            "photo_path": photo_path,
            "updated_at": int(time.time()),
        }

        self.sync._request("PATCH", self._profile_path(), record)
        return photo_path

    def clear_photo_path(self):
        record = {
            "photo_path": None,
            "updated_at": int(time.time()),
        }

        self.sync._request("PATCH", self._profile_path(), record)
        return True
