"""Gestao de sessoes Firebase do AlphaTrainer."""

import time
from cloud.firebase_auth import FirebaseAuth, FirebaseAuthError


class SessionManager:
    def __init__(self, auth=None, session=None, storage=None):
        self.auth = auth or FirebaseAuth()
        self.session = session
        self.storage = storage

    @property
    def is_logged_in(self):
        return bool(
            self.session
            and self.session.get("id_token")
            and self.session.get("refresh_token")
        )

    def start(self, session):
        if not session.get("id_token") or not session.get("refresh_token"):
            raise FirebaseAuthError("Sessao incompleta.")

        self.session = dict(session)
        self._save()
        return self.session

    def restore(self):
        if self.storage is None:
            return None

        saved = self.storage.load()
        if not saved:
            return None

        self.session = saved
        return self.ensure_valid()

    def ensure_valid(self):
        if not self.is_logged_in:
            raise FirebaseAuthError("Utilizador sem sessao ativa.")

        expires_at = float(self.session.get("expires_at", 0))

        if expires_at <= time.time() + 300:
            refreshed = self.auth.refresh_session(
                self.session["refresh_token"]
            )
            self.session.update(refreshed)
            self._save()

        return self.session

    def get_id_token(self):
        return self.ensure_valid()["id_token"]

    def update(self, new_session):
        if not self.session:
            return self.start(new_session)

        self.session.update(new_session)
        self._save()
        return self.session

    def logout(self):
        self.session = None
        if self.storage is not None:
            self.storage.clear()

    def _save(self):
        if self.storage is not None and self.session is not None:
            self.storage.save(self.session)
