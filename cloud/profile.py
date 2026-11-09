"""Gestao do perfil do utilizador AlphaTrainer."""


class UserProfile:
    def __init__(self, auth, session):
        self.auth = auth
        self.session = session
        self.data = {}

    def refresh(self):
        account = self.auth.get_account(self.session["id_token"])
        self.data = {
            "uid": account.get("localId"),
            "name": account.get("displayName", ""),
            "email": account.get("email", ""),
            "photo_url": account.get("photoUrl", ""),
            "email_verified": account.get("emailVerified", False),
            "created_at": account.get("createdAt"),
            "last_login": account.get("lastLoginAt"),
            "provider": self._provider(account),
        }
        return self.data

    @staticmethod
    def _provider(account):
        providers = account.get("providerUserInfo", [])
        if any(p.get("providerId") == "google.com" for p in providers):
            return "Google"
        if any(p.get("providerId") == "password" for p in providers):
            return "Email e palavra-passe"
        return "Outro"

    def update(self, name=None, photo_url=None):
        result = self.auth.update_profile(
            self.session["id_token"],
            display_name=name,
            photo_url=photo_url,
        )
        self.session.update(self.auth._session(result))
        return self.refresh()
