"""Sincronizacao dos modelos AlphaTrainer com Firebase RTDB."""

import base64
import json
import os
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


DATABASE_URL = (
    "https://allphatrainer-default-rtdb.firebaseio.com"
)


class FirebaseSyncError(Exception):
    pass


class FirebaseSync:
    def __init__(
        self, session, database_url=DATABASE_URL, timeout=20,
        session_manager=None
    ):
        self.session_manager = session_manager
        self.session = (
            session_manager.session
            if session_manager is not None
            else session
        ) or {}
        self.database_url = database_url.rstrip("/")
        self.timeout = timeout

    def _url(self, path):
        if self.session_manager is not None:
            token = self.session_manager.get_id_token()
            self.session = self.session_manager.session or {}
        else:
            token = self.session.get("id_token")

        if not token:
            raise FirebaseSyncError("Sessao Firebase inexistente.")

        return (
            f"{self.database_url}/{path}.json?"
            f"{urlencode({'auth': token})}"
        )

    def _request(self, method, path, data=None):
        body = None if data is None else json.dumps(data).encode("utf-8")

        request = Request(
            self._url(path),
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else None

        except HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = str(exc)

            raise FirebaseSyncError(
                f"Firebase respondeu HTTP {exc.code}: {detail[:300]}"
            ) from exc

        except (URLError, TimeoutError, OSError) as exc:
            raise FirebaseSyncError(
                "Falha de ligacao ao Firebase."
            ) from exc

    def _model_path(self, name):
        uid = self.session.get("uid")
        if not uid:
            raise FirebaseSyncError("UID do utilizador inexistente.")

        encoded = base64.urlsafe_b64encode(
            name.encode("utf-8")
        ).decode("ascii").rstrip("=")

        return f"users/{quote(uid, safe='')}/models/{encoded}"

    def upload_model(self, alpha, name):
        """Envia um modelo local para a nuvem."""
        if name not in alpha.list_models():
            raise FirebaseSyncError(f"Modelo local inexistente: {name}")

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)

        try:
            alpha.save_model(name, path)

            with open(path, "r", encoding="utf-8") as file:
                checkpoint = json.load(file)

            record = {
                "name": name,
                "checkpoint": checkpoint,
                "updated_at": int(time.time()),
            }

            self._request("PUT", self._model_path(name), record)
            return True

        finally:
            if os.path.exists(path):
                os.remove(path)

    def download_models(self, alpha):
        """Importa modelos da nuvem sem apagar os modelos locais."""
        uid = self.session.get("uid")
        if not uid:
            raise FirebaseSyncError("UID do utilizador inexistente.")

        records = self._request(
            "GET",
            f"users/{quote(uid, safe='')}/models"
        ) or {}

        imported = []

        for record in records.values():
            if not isinstance(record, dict):
                continue

            name = record.get("name")
            checkpoint = record.get("checkpoint")

            if not isinstance(name, str) or not isinstance(checkpoint, dict):
                continue

            fd, path = tempfile.mkstemp(suffix=".json")
            os.close(fd)

            try:
                with open(path, "w", encoding="utf-8") as file:
                    json.dump(checkpoint, file)

                if name in alpha.list_models():
                    continue

                alpha.load_model(name, path)
                imported.append(name)

            finally:
                if os.path.exists(path):
                    os.remove(path)

        return imported

    def sync_models(self, alpha):
        """Envia os modelos locais e importa os modelos remotos."""
        uploaded = []

        for name in alpha.list_models():
            self.upload_model(alpha, name)
            uploaded.append(name)

        downloaded = self.download_models(alpha)

        return {
            "uploaded": uploaded,
            "downloaded": downloaded,
            "synced_at": int(time.time()),
        }
