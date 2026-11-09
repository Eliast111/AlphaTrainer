import unittest
import time

from cloud.session import SessionManager


class FakeAuth:
    def __init__(self):
        self.calls = 0

    def refresh_session(self, refresh_token):
        self.calls += 1
        assert refresh_token == "refresh_antigo"

        return {
            "id_token": "token_novo",
            "refresh_token": "refresh_novo",
            "expires_at": time.time() + 3600,
        }


class FakeStorage:
    def __init__(self, data=None):
        self.data = data
        self.saved = None
        self.cleared = False

    def load(self):
        return self.data

    def save(self, session):
        self.saved = dict(session)

    def clear(self):
        self.data = None
        self.saved = None
        self.cleared = True


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.session = {
            "uid": "utilizador_teste",
            "id_token": "token_antigo",
            "refresh_token": "refresh_antigo",
            "expires_at": time.time() - 10,
        }

    def test_start_and_login_status(self):
        manager = SessionManager(FakeAuth())
        manager.start(self.session)

        self.assertTrue(manager.is_logged_in)

    def test_expired_token_is_refreshed(self):
        auth = FakeAuth()
        manager = SessionManager(auth)
        manager.start(self.session)

        token = manager.get_id_token()

        self.assertEqual(token, "token_novo")
        self.assertEqual(auth.calls, 1)
        self.assertEqual(manager.session["refresh_token"], "refresh_novo")

    def test_restore_and_refresh(self):
        auth = FakeAuth()
        storage = FakeStorage(dict(self.session))
        manager = SessionManager(auth, storage=storage)

        restored = manager.restore()

        self.assertEqual(restored["id_token"], "token_novo")
        self.assertEqual(storage.saved["refresh_token"], "refresh_novo")

    def test_logout_clears_session(self):
        storage = FakeStorage()
        manager = SessionManager(FakeAuth(), storage=storage)
        manager.start({
            **self.session,
            "expires_at": time.time() + 3600,
        })

        manager.logout()

        self.assertFalse(manager.is_logged_in)
        self.assertTrue(storage.cleared)

    def test_restore_without_saved_session(self):
        manager = SessionManager(FakeAuth(), storage=FakeStorage())

        self.assertIsNone(manager.restore())
        self.assertFalse(manager.is_logged_in)


if __name__ == "__main__":
    unittest.main()
