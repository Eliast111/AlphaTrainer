import sys
from pathlib import Path
import getpass
import runpy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cloud.firebase_auth import FirebaseAuth

config = runpy.run_path(
    str(ROOT / "android_app" / "firebase_config.py")
)

api_key = config.get("FIREBASE_API_KEY")

if not api_key:
    raise SystemExit("API Key não encontrada.")

email = input("Email: ").strip()
password = getpass.getpass("Palavra-passe: ")
name = input("Nome: ").strip()

if "@" not in email or "." not in email.split("@")[-1]:
    raise SystemExit("Email inválido.")

if len(password) < 6:
    raise SystemExit("Palavra-passe demasiado curta.")

try:
    auth = FirebaseAuth(api_key=api_key)
    session = auth.register(email, password, name)

    print("\nCONTA CRIADA COM SUCESSO")
    print("UID:", session["uid"])
    print("Email:", session["email"])

except Exception as exc:
    print("\nERRO FIREBASE:", str(exc))
    raise SystemExit(1)
