"""Prepare the Android source without copying private local credentials."""
from pathlib import Path
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
ANDROID = Path(__file__).resolve().parent
DEST = ANDROID / "src"

PACKAGES = [
    "alpha_math", "checkpoints", "core", "data", "evaluation",
    "language", "learning", "models", "neural", "registry",
    "training", "transformer", "cloud"
]

DEST.mkdir(exist_ok=True)

for package in PACKAGES:
    source = ROOT / package
    target = DEST / package

    if target.exists():
        shutil.rmtree(target)

    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.bak*", "*.pre-fix"
        )
    )

shutil.copy2(ANDROID / "main.py", DEST / "main.py")

# Nunca copiar firebase_config.py local.
firebase_key = os.environ.get("FIREBASE_API_KEY", "")
supabase_url = os.environ.get("SUPABASE_URL", "")
supabase_key = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "")

config = (
    '"""Configuracao gerada para a compilacao Android."""\n'
    f"FIREBASE_API_KEY = {firebase_key!r}\n"
    f"SUPABASE_URL = {supabase_url!r}\n"
    f"SUPABASE_PUBLISHABLE_KEY = {supabase_key!r}\n"
)

(DEST / "firebase_config.py").write_text(config, encoding="utf-8")

print(f"Codigo preparado em: {DEST}")
print("Configuracao Android gerada.")
