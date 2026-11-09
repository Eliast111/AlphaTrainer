import unittest
from unittest.mock import patch

from cloud.firebase_auth import FirebaseAuth, FirebaseAuthError


class TestFirebaseAuth(unittest.TestCase):

    def setUp(self):
        self.auth = FirebaseAuth(api_key="test-key")

    def fake_session(self):
        return {
            "localId": "user-123",
            "email": "teste@example.com",
            "idToken": "fake-id-token",
            "refreshToken": "fake-refresh-token",
            "expiresIn": "3600",
            "emailVerified": False,
        }

    @patch.object(FirebaseAuth, "_identity")
    def test_register(self, identity):
        identity.return_value = self.fake_session()

        result = self.auth.register(
            "teste@example.com", "password123"
        )

        self.assertEqual(result["uid"], "user-123")
        self.assertEqual(result["email"], "teste@example.com")
        self.assertEqual(result["refresh_token"], "fake-refresh-token")

    @patch.object(FirebaseAuth, "_identity")
    def test_login(self, identity):
        identity.return_value = self.fake_session()

        result = self.auth.login("teste@example.com", "password123")

        self.assertEqual(result["uid"], "user-123")
        self.assertTrue(result["id_token"])

    @patch.object(FirebaseAuth, "_identity")
    def test_password_reset(self, identity):
        identity.return_value = {"email": "teste@example.com"}

        result = self.auth.send_password_reset("teste@example.com")

        self.assertIn("email", result)

    @patch.object(FirebaseAuth, "_identity")
    def test_email_verification(self, identity):
        identity.return_value = {"email": "teste@example.com"}

        result = self.auth.send_email_verification("fake-token")

        self.assertIn("email", result)

    @patch.object(FirebaseAuth, "_post")
    def test_refresh_session(self, post):
        post.return_value = {
            "user_id": "user-123",
            "id_token": "new-id-token",
            "refresh_token": "new-refresh-token",
            "expires_in": "3600",
        }

        result = self.auth.refresh_session("fake-refresh-token")

        self.assertEqual(result["uid"], "user-123")
        self.assertEqual(result["id_token"], "new-id-token")

    @patch.object(FirebaseAuth, "_identity")
    def test_get_account(self, identity):
        identity.return_value = {
            "users": [{"localId": "user-123"}]
        }

        result = self.auth.get_account("fake-token")

        self.assertEqual(result["localId"], "user-123")

    @patch.object(FirebaseAuth, "_identity")
    def test_update_profile(self, identity):
        identity.return_value = self.fake_session()

        result = self.auth.update_profile(
            "fake-token", display_name="Alpha User"
        )

        self.assertTrue(result["idToken"])
        self.assertEqual(identity.call_args.args[0], "accounts:update")

    def test_missing_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(FirebaseAuthError):
                FirebaseAuth()


if __name__ == "__main__":
    unittest.main()
