import json
import tempfile
import unittest
from unittest.mock import patch

from cloud.firebase_sync import FirebaseSync, FirebaseSyncError


class FakeAlpha:
    def __init__(self):
        self.models = {"modelo_local": {"weights": [1, 2]}}
        self.loaded = []

    def list_models(self):
        return list(self.models)

    def save_model(self, name, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.models[name], f)

    def load_model(self, name, path):
        with open(path, encoding="utf-8") as f:
            self.models[name] = json.load(f)
        self.loaded.append(name)


class TestFirebaseSync(unittest.TestCase):
    def setUp(self):
        self.session = {"uid": "uid_teste", "id_token": "token_teste"}
        self.alpha = FakeAlpha()
        self.sync = FirebaseSync(self.session)

    def test_model_path_isolated_by_uid(self):
        path = self.sync._model_path("modelo local")
        self.assertTrue(path.startswith("users/uid_teste/models/"))
        self.assertNotIn("token_teste", path)

    def test_missing_uid_is_rejected(self):
        sync = FirebaseSync({"id_token": "token"})
        with self.assertRaises(FirebaseSyncError):
            sync._model_path("modelo")

    def test_upload_model(self):
        with patch.object(self.sync, "_request", return_value="ok") as request:
            result = self.sync.upload_model(self.alpha, "modelo_local")

        self.assertTrue(result)
        request.assert_called_once()
        method, path, record = request.call_args.args
        self.assertEqual(method, "PUT")
        self.assertTrue(path.startswith("users/uid_teste/models/"))
        self.assertEqual(record["name"], "modelo_local")
        self.assertEqual(record["checkpoint"]["weights"], [1, 2])

    def test_upload_unknown_model_fails(self):
        with self.assertRaises(FirebaseSyncError):
            self.sync.upload_model(self.alpha, "nao_existe")

    def test_download_preserves_existing_local_model(self):
        records = {
            "remoto": {
                "name": "modelo_remoto",
                "checkpoint": {"weights": [3, 4]},
            },
            "local": {
                "name": "modelo_local",
                "checkpoint": {"weights": [9, 9]},
            },
        }
        with patch.object(self.sync, "_request", return_value=records):
            imported = self.sync.download_models(self.alpha)

        self.assertEqual(imported, ["modelo_remoto"])
        self.assertEqual(self.alpha.models["modelo_local"]["weights"], [1, 2])
        self.assertEqual(self.alpha.models["modelo_remoto"]["weights"], [3, 4])

    def test_download_without_models(self):
        with patch.object(self.sync, "_request", return_value=None):
            imported = self.sync.download_models(self.alpha)
        self.assertEqual(imported, [])

    def test_sync_report(self):
        with patch.object(self.sync, "upload_model", return_value=True) as upload, \
             patch.object(self.sync, "download_models", return_value=["novo"]):
            result = self.sync.sync_models(self.alpha)

        upload.assert_called_once_with(self.alpha, "modelo_local")
        self.assertEqual(result["uploaded"], ["modelo_local"])
        self.assertEqual(result["downloaded"], ["novo"])
        self.assertIn("synced_at", result)

    def test_download_without_uid_fails(self):
        sync = FirebaseSync({"id_token": "token"})
        with self.assertRaises(FirebaseSyncError):
            sync.download_models(self.alpha)


    def test_sync_uses_refreshed_session_token(self):
        class Manager:
            def __init__(self):
                self.session = {
                    "uid": "uid_teste",
                    "id_token": "token_antigo",
                }
                self.calls = 0

            def get_id_token(self):
                self.calls += 1
                self.session["id_token"] = "token_novo"
                return "token_novo"

        manager = Manager()
        sync = FirebaseSync(
            manager.session,
            session_manager=manager
        )

        url = sync._url("users/uid_teste/models")

        self.assertIn("auth=token_novo", url)
        self.assertEqual(manager.calls, 1)
        self.assertEqual(sync.session["id_token"], "token_novo")

    def test_sync_without_session_manager_keeps_compatibility(self):
        sync = FirebaseSync(self.session)
        url = sync._url("users/uid_teste/models")
        self.assertIn("auth=token_teste", url)


if __name__ == "__main__":
    unittest.main()
