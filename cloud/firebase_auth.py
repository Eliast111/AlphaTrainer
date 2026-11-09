"""Firebase Authentication do AlphaTrainer."""

import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


IDENTITY_URL = "https://identitytoolkit.googleapis.com/v1"
TOKEN_URL = "https://securetoken.googleapis.com/v1/token"


class FirebaseAuthError(Exception):
    pass


class FirebaseAuth:
    def __init__(self, api_key=None, timeout=15):
        self.api_key = api_key or os.environ.get(
            "ALPHATRAINER_FIREBASE_API_KEY"
        )

        if not self.api_key:
            raise FirebaseAuthError(
                "Define ALPHATRAINER_FIREBASE_API_KEY."
            )

        self.timeout = timeout

    def _post(self, url, data):
        request = Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))

        except HTTPError as exc:
            try:
                result = json.loads(exc.read().decode("utf-8"))
                message = result.get("error", {}).get(
                    "message", "Erro de autenticacao"
                )
            except (ValueError, AttributeError):
                message = "Erro HTTP na autenticacao"

            raise FirebaseAuthError(message) from exc

        except (URLError, TimeoutError, OSError) as exc:
            raise FirebaseAuthError(
                "Nao foi possivel contactar o Firebase."
            ) from exc

    def _identity(self, action, data):
        url = (
            f"{IDENTITY_URL}/{action}?"
            f"{urlencode({'key': self.api_key})}"
        )
        return self._post(url, data)

    @staticmethod
    def _session(result):
        return {
            "uid": result.get("localId") or result.get("user_id"),
            "email": result.get("email"),
            "id_token": result.get("idToken") or result.get("id_token"),
            "refresh_token": (
                result.get("refreshToken") or result.get("refresh_token")
            ),
            "expires_at": time.time() + int(
                result.get("expiresIn") or result.get("expires_in") or 3600
            ),
            "email_verified": result.get("emailVerified", False),
        }

    def register(self, email, password, display_name=""):
        result = self._identity("accounts:signUp", {
            "email": email.strip(),
            "password": password,
            "returnSecureToken": True,
        })

        session = self._session(result)

        if display_name.strip():
            result = self.update_profile(
                session["id_token"],
                display_name=display_name.strip()
            )
            session = self._session(result)

        return session

    def login(self, email, password):
        result = self._identity("accounts:signInWithPassword", {
            "email": email.strip(),
            "password": password,
            "returnSecureToken": True,
        })
        return self._session(result)

    def send_password_reset(self, email):
        return self._identity("accounts:sendOobCode", {
            "requestType": "PASSWORD_RESET",
            "email": email.strip(),
        })

    def send_email_verification(self, id_token):
        return self._identity("accounts:sendOobCode", {
            "requestType": "VERIFY_EMAIL",
            "idToken": id_token,
        })

    def refresh_session(self, refresh_token):
        url = f"{TOKEN_URL}?{urlencode({'key': self.api_key})}"
        result = self._post(url, {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        })
        return self._session(result)

    def get_account(self, id_token):
        result = self._identity("accounts:lookup", {
            "idToken": id_token,
        })

        users = result.get("users", [])
        if not users:
            raise FirebaseAuthError("Conta nao encontrada.")

        return users[0]

    def update_profile(
        self, id_token, display_name=None, photo_url=None
    ):
        data = {
            "idToken": id_token,
            "returnSecureToken": True,
        }

        if display_name is not None:
            data["displayName"] = display_name.strip()

        if photo_url is not None:
            data["photoUrl"] = photo_url

        return self._identity("accounts:update", data)
