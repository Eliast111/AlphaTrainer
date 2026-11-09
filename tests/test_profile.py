import unittest
from unittest.mock import Mock

from cloud.profile import UserProfile
from cloud.firebase_auth import FirebaseAuth


class TestUserProfile(unittest.TestCase):

    def setUp(self):
        self.auth = Mock()
        self.auth._session = FirebaseAuth._session
        self.session = {
            "id_token": "fake-id-token",
            "refresh_token": "fake-refresh-token",
        }
        self.profile = UserProfile(self.auth, self.session)

    def test_refresh_profile(self):
        self.auth.get_account.return_value = {
            "localId": "user-123",
            "displayName": "Alpha User",
            "email": "alpha@example.com",
            "photoUrl": "https://example.com/photo.jpg",
            "emailVerified": True,
            "createdAt": "2026-01-01",
            "lastLoginAt": "2026-11-04",
            "providerUserInfo": [{"providerId": "password"}],
        }

        data = self.profile.refresh()

        self.assertEqual(data["uid"], "user-123")
        self.assertEqual(data["name"], "Alpha User")
        self.assertTrue(data["email_verified"])
        self.assertEqual(data["provider"], "Email e palavra-passe")

    def test_google_provider(self):
        account = {
            "providerUserInfo": [{"providerId": "google.com"}]
        }
        self.assertEqual(
            self.profile._provider(account), "Google"
        )

    def test_other_provider(self):
        self.assertEqual(
            self.profile._provider({}), "Outro"
        )

    def test_update_profile(self):
        self.auth.update_profile.return_value = {
            "localId": "user-123",
            "email": "alpha@example.com",
            "idToken": "new-id-token",
            "refreshToken": "new-refresh-token",
            "expiresIn": "3600",
        }

        self.auth.get_account.return_value = {
            "localId": "user-123",
            "displayName": "Novo Nome",
            "email": "alpha@example.com",
            "providerUserInfo": [{"providerId": "password"}],
        }

        data = self.profile.update(name="Novo Nome")

        self.assertEqual(data["name"], "Novo Nome")
        self.assertEqual(self.session["id_token"], "new-id-token")


if __name__ == "__main__":
    unittest.main()
