"""Servico de fotografias de perfil do AlphaTrainer."""

import json
import os
import re
import time
from pathlib import Path

from cloud.supabase_storage import SupabaseStorage
from cloud.profile_photo_metadata import ProfilePhotoMetadata


class ProfilePhotoServiceError(Exception):
    pass


class ProfilePhotoService:
    MAX_SIZE = 5 * 1024 * 1024

    SIGNATURES = {
        "jpg": lambda b: b.startswith(b"\xff\xd8\xff"),
        "png": lambda b: b.startswith(
            b"\x89PNG\r\n\x1a\n"
        ),
        "webp": lambda b: (
            len(b) >= 12
            and b[:4] == b"RIFF"
            and b[8:12] == b"WEBP"
        ),
    }

    def __init__(self, session_manager, cache_dir):
        self.session_manager = session_manager
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.storage = SupabaseStorage(
            session_manager=session_manager
        )

    def _session(self):
        return self.session_manager.ensure_valid()

    def _uid(self):
        session = self._session()
        uid = session.get("uid")

        if not isinstance(uid, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", uid
        ):
            raise ProfilePhotoServiceError("UID invalido.")

        return session, uid

    def _metadata(self, session):
        return ProfilePhotoMetadata(
            session,
            session_manager=self.session_manager,
        )

    def _detect_extension(self, data):
        if not isinstance(data, bytes) or not data:
            raise ProfilePhotoServiceError("Imagem vazia ou invalida.")

        if len(data) > self.MAX_SIZE:
            raise ProfilePhotoServiceError(
                "A imagem ultrapassa o limite de 5 MB."
            )

        for extension, check in self.SIGNATURES.items():
            if check(data):
                return extension

        raise ProfilePhotoServiceError(
            "Formato nao permitido. Usa JPEG, PNG ou WEBP."
        )

    def _user_dir(self, uid):
        path = self.cache_dir / uid
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _pointer(self, uid):
        return self._user_dir(uid) / "profile_photo.json"

    def _save_pointer(self, uid, photo_path):
        target = self._pointer(uid)
        temporary = target.with_suffix(".tmp")

        temporary.write_text(
            json.dumps({
                "photo_path": photo_path,
                "updated_at": int(time.time()),
            }),
            encoding="utf-8",
        )
        os.replace(temporary, target)

    def _read_pointer(self, uid):
        target = self._pointer(uid)

        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            photo_path = data.get("photo_path")
        except (OSError, ValueError, AttributeError):
            return None

        if (
            not isinstance(photo_path, str)
            or not re.fullmatch(
                rf"{re.escape(uid)}/avatar\.(jpg|jpeg|png|webp)",
                photo_path,
            )
        ):
            return None

        return photo_path

    def _local_file(self, uid, photo_path):
        filename = photo_path.split("/", 1)[1]
        return self._user_dir(uid) / filename

    def upload_bytes(self, data):
        session, uid = self._uid()
        extension = self._detect_extension(data)

        local_file = self._user_dir(uid) / f"avatar.{extension}"
        temporary = local_file.with_suffix(local_file.suffix + ".tmp")

        temporary.write_bytes(data)
        os.replace(temporary, local_file)

        result = self.storage.upload_photo(uid, local_file)
        photo_path = result["photo_path"]

        self._metadata(session).set_photo_path(photo_path)
        self._save_pointer(uid, photo_path)

        return str(local_file)

    def load_photo(self):
        try:
            session, uid = self._uid()
        except Exception:
            session = getattr(self.session_manager, "session", None)
            if not isinstance(session, dict):
                raise ProfilePhotoServiceError(
                    "Nao foi possivel recuperar a sessao."
                )
            uid = session.get("uid")
            if not isinstance(uid, str) or not re.fullmatch(
                r"[A-Za-z0-9_-]+", uid
            ):
                raise ProfilePhotoServiceError("UID invalido.")

        try:
            photo_path = self._metadata(session).get_photo_path()
            if photo_path:
                self._save_pointer(uid, photo_path)
        except Exception:
            photo_path = self._read_pointer(uid)

        if not photo_path:
            return None

        local_file = self._local_file(uid, photo_path)

        if local_file.is_file():
            if 0 < local_file.stat().st_size <= self.MAX_SIZE:
                return str(local_file)

        self.storage.download_photo(
            uid, photo_path, local_file
        )
        return str(local_file)

    def clear_photo(self):
        session, uid = self._uid()
        self._metadata(session).clear_photo_path()

        pointer = self._pointer(uid)
        try:
            pointer.unlink()
        except FileNotFoundError:
            pass

        return True
