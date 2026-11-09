"""Armazenamento privado de fotografias do AlphaTrainer."""

import mimetypes
import os
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class SupabaseStorageError(Exception):
    pass


class SupabaseStorage:
    BUCKET = "profile-photos"
    MAX_SIZE = 5 * 1024 * 1024
    ALLOWED_TYPES = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }

    def __init__(
        self, url=None, publishable_key=None,
        session_manager=None, timeout=25
    ):
        if not url or not publishable_key:
            try:
                from firebase_config import (
                    SUPABASE_URL,
                    SUPABASE_PUBLISHABLE_KEY,
                )
            except ImportError:
                try:
                    from android_app.firebase_config import (
                        SUPABASE_URL,
                        SUPABASE_PUBLISHABLE_KEY,
                    )
                except ImportError:
                    SUPABASE_URL = ""
                    SUPABASE_PUBLISHABLE_KEY = ""

            url = url or os.environ.get("SUPABASE_URL") or SUPABASE_URL
            publishable_key = (
                publishable_key
                or os.environ.get("SUPABASE_PUBLISHABLE_KEY")
                or SUPABASE_PUBLISHABLE_KEY
            )

        self.url = (url or "").rstrip("/")
        self.key = publishable_key or ""
        self.session_manager = session_manager
        self.timeout = timeout

        if not self.url or not self.key:
            raise SupabaseStorageError(
                "Configuracao Supabase incompleta."
            )

        if not self.url.startswith("https://"):
            raise SupabaseStorageError(
                "A URL Supabase deve utilizar HTTPS."
            )

    def _headers(self):
        if self.session_manager is None:
            raise SupabaseStorageError(
                "Sessao Firebase inexistente."
            )

        session = self.session_manager.ensure_valid()
        token = session.get("id_token")

        if not token:
            raise SupabaseStorageError(
                "Token Firebase inexistente."
            )

        return {
            "apikey": self.key,
            "Authorization": f"Bearer {token}",
        }

    @staticmethod
    def _safe_uid(uid):
        if not isinstance(uid, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", uid
        ):
            raise SupabaseStorageError("UID invalido.")
        return uid

    def _request(self, method, endpoint, data=None, headers=None):
        request_headers = self._headers()
        if headers:
            request_headers.update(headers)

        request = Request(
            self.url + endpoint,
            data=data,
            headers=request_headers,
            method=method,
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.read()

        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise SupabaseStorageError(
                f"Supabase HTTP {exc.code}: {detail[:250]}"
            ) from exc

        except (URLError, TimeoutError, OSError) as exc:
            raise SupabaseStorageError(
                "Falha de ligacao ao Supabase."
            ) from exc

    def upload_photo(self, uid, file_path):
        uid = self._safe_uid(uid)
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension not in self.ALLOWED_TYPES:
            raise SupabaseStorageError(
                "Formato permitido: JPG, PNG ou WEBP."
            )

        if not path.is_file():
            raise SupabaseStorageError(
                "Ficheiro de fotografia inexistente."
            )

        size = path.stat().st_size
        if size <= 0 or size > self.MAX_SIZE:
            raise SupabaseStorageError(
                "A fotografia deve ter entre 1 byte e 5 MB."
            )

        content_type = self.ALLOWED_TYPES[extension]
        object_path = f"{uid}/avatar{extension}"

        encoded_path = quote(object_path, safe="/")
        endpoint = (
            f"/storage/v1/object/{self.BUCKET}/{encoded_path}"
        )

        with path.open("rb") as image_file:
            data = image_file.read(self.MAX_SIZE + 1)

        if len(data) > self.MAX_SIZE:
            raise SupabaseStorageError(
                "A fotografia ultrapassa 5 MB."
            )

        self._request(
            "POST",
            endpoint,
            data=data,
            headers={
                "Content-Type": content_type,
                "x-upsert": "true",
            },
        )

        return {
            "photo_path": object_path,
            "content_type": content_type,
            "size": len(data),
        }

    def download_photo(self, uid, photo_path, destination):
        uid = self._safe_uid(uid)

        if not isinstance(photo_path, str):
            raise SupabaseStorageError("Caminho invalido.")

        expected_prefix = uid + "/"
        if not photo_path.startswith(expected_prefix):
            raise SupabaseStorageError(
                "A fotografia nao pertence a este utilizador."
            )

        filename = photo_path[len(expected_prefix):]
        if filename not in (
            "avatar.jpg", "avatar.jpeg",
            "avatar.png", "avatar.webp"
        ):
            raise SupabaseStorageError(
                "Nome de fotografia invalido."
            )

        endpoint = (
            f"/storage/v1/object/authenticated/"
            f"{self.BUCKET}/{quote(photo_path, safe='/')}"
        )

        content = self._request("GET", endpoint)

        if not content or len(content) > self.MAX_SIZE:
            raise SupabaseStorageError(
                "Resposta de fotografia invalida."
            )

        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)

        temporary = target.with_name(target.name + ".tmp")
        try:
            temporary.write_bytes(content)
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()

        return str(target)
