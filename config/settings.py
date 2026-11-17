"""Configuracao central do AlphaTrainer."""

import os
from dataclasses import dataclass


DEFAULT_FIREBASE_DATABASE_URL = (
    "https://allphatrainer-default-rtdb.firebaseio.com"
)


@dataclass(frozen=True)
class Settings:
    firebase_api_key: str = ""
    firebase_database_url: str = DEFAULT_FIREBASE_DATABASE_URL
    supabase_url: str = ""
    supabase_publishable_key: str = ""

    @classmethod
    def from_env(cls):
        return cls(
            firebase_api_key=os.environ.get(
                "ALPHATRAINER_FIREBASE_API_KEY", ""
            ).strip(),
            firebase_database_url=os.environ.get(
                "ALPHATRAINER_FIREBASE_DATABASE_URL",
                DEFAULT_FIREBASE_DATABASE_URL,
            ).strip().rstrip("/"),
            supabase_url=os.environ.get(
                "ALPHATRAINER_SUPABASE_URL",
                os.environ.get("SUPABASE_URL", ""),
            ).strip().rstrip("/"),
            supabase_publishable_key=os.environ.get(
                "ALPHATRAINER_SUPABASE_PUBLISHABLE_KEY",
                os.environ.get("SUPABASE_PUBLISHABLE_KEY", ""),
            ).strip(),
        )
