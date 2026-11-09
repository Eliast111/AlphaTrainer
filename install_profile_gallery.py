from pathlib import Path
from datetime import datetime
import shutil

p = Path("android_app/main.py")
s = p.read_text(encoding="utf-8")

# Evitar instalar duas vezes
if "def _open_profile_gallery(self" in s:
    raise SystemExit("A integracao da galeria ja existe.")

# Confirmar todos os pontos de insercao antes de alterar
anchors = [
    "from kivy.uix.scrollview import ScrollView",
    "from cloud.firebase_sync import FirebaseSync",
    "        self.root = BoxLayout(orientation=\"vertical\")",
    "    def update_profile(self, *_):",
    "        def operation():\n            session = self.session_manager.ensure_valid()\n            return UserProfile(self.auth, session).refresh()",
    "        def success(profile):\n            loading.text = (",
    "            page.add_widget(self.profile_name_input)",
]

for anchor in anchors:
    if anchor not in s:
        raise SystemExit(
            "Ponto de insercao nao encontrado. Nada foi alterado:\n"
            + repr(anchor)
        )

backup_dir = Path("backups")
backup_dir.mkdir(exist_ok=True)
backup = backup_dir / "main.py.before_gallery"
if backup.exists():
    backup = backup_dir / (
        "main.py.before_gallery."
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
shutil.copy2(p, backup)

# Imports
s = s.replace(
    "from kivy.uix.scrollview import ScrollView",
    "from kivy.uix.scrollview import ScrollView\n"
    "from kivy.uix.image import Image\n"
    "from kivy.utils import platform",
    1
)
s = s.replace(
    "from cloud.firebase_sync import FirebaseSync",
    "from cloud.firebase_sync import FirebaseSync\n"
    "from cloud.profile_photo_service import ProfilePhotoService",
    1
)

# Registo do seletor Android
s = s.replace(
    '        self.root = BoxLayout(orientation="vertical")',
    '''        self._gallery_request_code = 7412
        self._gallery_activity = None

        if platform == "android":
            try:
                from android import activity
                activity.bind(
                    on_activity_result=self._on_gallery_result
                )
                self._gallery_activity = activity
            except Exception as exc:
                print("Seletor Android indisponivel:", exc)

        self.root = BoxLayout(orientation="vertical")''',
    1
)

# Imagem no perfil
s = s.replace(
    '        page.add_widget(loading)\n\n        self.add_status(page)',
    '''        self.profile_photo_image = Image(
            source="",
            size_hint=(None, None),
            size=(dp(130), dp(130)),
            allow_stretch=True,
            keep_ratio=True,
        )
        page.add_widget(self.profile_photo_image)
        page.add_widget(loading)

        self.add_status(page)''',
    1
)

# Carregamento do perfil e da fotografia
old_operation = '''        def operation():
            session = self.session_manager.ensure_valid()
            return UserProfile(self.auth, session).refresh()'''

new_operation = '''        def operation():
            session = self.session_manager.ensure_valid()
            profile = UserProfile(self.auth, session).refresh()

            photo_service = ProfilePhotoService(
                self.session_manager,
                Path(self.user_data_dir) / "profile_photos",
            )

            try:
                photo_path = photo_service.load_photo()
            except Exception as exc:
                print("Fotografia ainda nao carregada:", exc)
                photo_path = None

            return profile, photo_path, photo_service'''

s = s.replace(old_operation, new_operation, 1)

s = s.replace(
    '        def success(profile):\n            loading.text = (',
    '''        def success(result):
            profile, photo_path, photo_service = result
            self.photo_service = photo_service

            if photo_path:
                self.profile_photo_image.source = photo_path
                self.profile_photo_image.reload()

            loading.text = (''',
    1
)

s = s.replace(
    "            page.add_widget(self.profile_name_input)",
    '''            page.add_widget(self.profile_name_input)

            self.add_button(
                page,
                "Alterar fotografia",
                self._open_profile_gallery,
            )''',
    1
)

# Metodos da galeria
gallery_methods = r'''
    def _open_profile_gallery(self, *_):
        if platform != "android":
            self.set_status(
                "A galeria nativa so esta disponivel no Android."
            )
            return

        if getattr(self, "_operation_busy", False):
            self.set_status("Aguarde a operacao em curso.")
            return

        try:
            from jnius import autoclass

            Intent = autoclass("android.content.Intent")
            PythonActivity = autoclass(
                "org.kivy.android.PythonActivity"
            )

            intent = Intent("android.intent.action.OPEN_DOCUMENT")
            intent.addCategory(
                "android.intent.category.OPENABLE"
            )
            intent.setType("image/*")
            intent.addFlags(1)

            PythonActivity.mActivity.startActivityForResult(
                intent, self._gallery_request_code
            )
            self.set_status("Escolha uma fotografia da galeria.")

        except Exception as exc:
            self.set_status("Erro ao abrir galeria: " + str(exc))

    def _read_gallery_uri(self, uri):
        from jnius import autoclass

        PythonActivity = autoclass(
            "org.kivy.android.PythonActivity"
        )
        activity = PythonActivity.mActivity
        resolver = activity.getContentResolver()
        stream = resolver.openInputStream(uri)

        if stream is None:
            raise RuntimeError("Nao foi possivel abrir a imagem.")

        data = bytearray()

        try:
            while True:
                value = stream.read()
                if value == -1:
                    break

                data.append(value)

                if len(data) > 5 * 1024 * 1024:
                    raise ValueError(
                        "A imagem ultrapassa o limite de 5 MB."
                    )
        finally:
            stream.close()

        return bytes(data)

    def _on_gallery_result(self, request_code, result_code, intent):
        if request_code != self._gallery_request_code:
            return

        if result_code != -1 or intent is None:
            self.set_status("Selecao cancelada.")
            return

        uri = intent.getData()
        if uri is None:
            self.set_status("Nenhuma imagem foi selecionada.")
            return

        def operation():
            data = self._read_gallery_uri(uri)
            return self.photo_service.upload_bytes(data)

        def success(local_path):
            self.profile_photo_image.source = local_path
            self.profile_photo_image.reload()
            self.set_status(
                "Fotografia atualizada com sucesso."
            )

        self._run_async(
            operation,
            success,
            "A validar e enviar fotografia...",
            "Erro na fotografia",
        )

'''

s = s.replace(
    "    def update_profile(self, *_):",
    gallery_methods + "    def update_profile(self, *_):",
    1
)

# Escrever apenas depois de todas as substituicoes
p.write_text(s, encoding="utf-8")

print("Integracao instalada.")
print("Backup:", backup)
print("Galeria Android ligada ao perfil.")
