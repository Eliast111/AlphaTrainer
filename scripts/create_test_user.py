import sys
from pathlib import Path
import getpass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cloud.firebase_auth import FirebaseAuth


email = input("Email: ").strip()
password = getpass.getpass("Palavra-passe: ")
name = input("Nome: ").strip()

if "@" not in email or "." not in email.split("@")[-1]:
    raise SystemExit("Email inválido.")

if len(password) < 6:
    raise SystemExit("Palavra-passe demasiado curta.")

try:
    auth = FirebaseAuth()
    session = auth.register(email, password, name)

    print("\nCONTA CRIADA COM SUCESSO")
    print("UID:", session["uid"])
    print("Email:", session["email"])

except Exception as exc:
    print("\nERRO FIREBASE:", str(exc))
    raise SystemExit(1)
