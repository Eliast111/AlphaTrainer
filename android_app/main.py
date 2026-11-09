"""Aplicacao Android AlphaTrainer: treino local e conta Firebase."""

import os
from pathlib import Path
import threading

from kivy.app import App
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image
from kivy.utils import platform

from core import AlphaCore
from data.samples import available_samples, load_sample
from learning.optimizer import create_optimizer

from cloud.firebase_auth import FirebaseAuth
from cloud.profile import UserProfile
from cloud.session import SessionManager
from cloud.firebase_sync import FirebaseSync
from cloud.profile_photo_service import ProfilePhotoService

try:
    from firebase_config import FIREBASE_API_KEY
except ImportError:
    FIREBASE_API_KEY = ""


class AlphaTrainerAndroid(App):
    def build(self):
        self.alpha = AlphaCore()
        self.auth = None
        self.session_manager = None

        api_key = FIREBASE_API_KEY or os.environ.get(
            "ALPHATRAINER_FIREBASE_API_KEY", ""
        )

        if api_key:
            try:
                self.auth = FirebaseAuth(api_key=api_key)

                storage = None
                from kivy.utils import platform

                if platform == "android":
                    from cloud.android_secure_storage import (
                        AndroidKeystoreSessionStorage
                    )
                    storage = AndroidKeystoreSessionStorage()

                self.session_manager = SessionManager(
                    auth=self.auth,
                    storage=storage
                )

                if storage is not None:
                    try:
                        restored = self.session_manager.restore()
                        if restored:
                            print("Sessao anterior restaurada.")
                    except Exception as exc:
                        print("Nao foi possivel restaurar a sessao:", exc)
            except Exception as exc:
                print("Firebase indisponivel:", exc)

        self._gallery_request_code = 7412
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

        self.root = BoxLayout(orientation="vertical")
        self.show_home()
        return self.root

    def make_page(self, title):
        self._page_token = getattr(self, '_page_token', 0) + 1
        self.root.clear_widgets()

        scroll = ScrollView()
        content = BoxLayout(
            orientation="vertical",
            padding=dp(18),
            spacing=dp(10),
            size_hint_y=None,
        )
        content.bind(minimum_height=content.setter("height"))

        content.add_widget(Label(
            text=f"[b]{title}[/b]",
            markup=True,
            font_size="22sp",
            size_hint_y=None,
            height=dp(55),
        ))

        scroll.add_widget(content)
        self.root.add_widget(scroll)
        return content

    def add_button(self, parent, text, callback):
        button = Button(
            text=text,
            size_hint_y=None,
            height=dp(48),
        )
        button.bind(on_release=callback)
        parent.add_widget(button)
        return button

    def add_status(self, parent):
        self.status = Label(
            text="Pronto.",
            size_hint_y=None,
            height=dp(65),
            halign="left",
            valign="middle",
        )
        self.status.bind(
            size=lambda obj, value: setattr(obj, "text_size", value)
        )
        parent.add_widget(self.status)

    def set_status(self, message):
        if hasattr(self, "status"):
            self.status.text = str(message)

    def show_home(self, *_):
        page = self.make_page("AlphaTrainer Studio")
        page.add_widget(Label(
            text="V1.2 | Inteligencia Artificial em Python",
            size_hint_y=None,
            height=dp(35),
        ))

        self.add_button(page, "Conta e perfil", self.show_account)

        page.add_widget(Label(
            text="TREINO LOCAL",
            size_hint_y=None,
            height=dp(35),
        ))

        self.sample = Spinner(
            text="xor",
            values=tuple(available_samples()),
            size_hint_y=None,
            height=dp(48),
        )
        page.add_widget(self.sample)

        self.add_button(page, "Treinar exemplo", self.train_demo)

        self.model_name = TextInput(
            hint_text="Nome do modelo",
            text="demo_xor",
            multiline=False,
            size_hint_y=None,
            height=dp(48),
        )
        page.add_widget(self.model_name)

        self.inputs = TextInput(
            hint_text="Entradas separadas por virgulas: 0,1",
            multiline=False,
            size_hint_y=None,
            height=dp(48),
        )
        page.add_widget(self.inputs)

        self.add_button(page, "Fazer previsao", self.predict)
        self.add_status(page)

    def train_demo(self, *_):
        if getattr(self, '_operation_busy', False):
            self.set_status('Aguarde: existe uma operacao em curso.')
            return
        try:
            key = self.sample.text
            dataset = load_sample(key)
            name = "demo_" + key

            if name in self.alpha.list_models():
                self.alpha.remove_model(name)

            output = (
                "sigmoid" if key in ("xor", "and", "or")
                else "linear"
            )

            self.alpha.create_model(
                name,
                [dataset.input_size, 4, dataset.target_size],
                activation="tanh",
                output_activation=output,
                seed=42,
            )

            result = self.alpha.train_model(
                name,
                dataset,
                create_optimizer("adam", 0.05),
                epochs=250,
                batch_size=1,
                shuffle=True,
                seed=42,
            )

            self.model_name.text = name
            self.set_status(
                f"Treino concluido!\n"
                f"Perda final: {result.get('final_loss', 0):.6f}"
            )
        except Exception as exc:
            self.set_status(f"Erro no treino: {exc}")

    def predict(self, *_):
        if getattr(self, '_operation_busy', False):
            self.set_status('Aguarde: existe uma operacao em curso.')
            return
        try:
            name = self.model_name.text.strip()
            values = [
                float(value.strip())
                for value in self.inputs.text.split(",")
            ]
            result = self.alpha.predict(name, values)
            self.set_status(
                "Previsao: " +
                ", ".join(f"{value:.5f}" for value in result)
            )
        except Exception as exc:
            self.set_status(f"Erro na previsao: {exc}")

    def show_account(self, *_):
        page = self.make_page("Conta AlphaTrainer")

        self.add_button(page, "Voltar ao treino", self.show_home)

        if self.auth is None:
            page.add_widget(Label(
                text=(
                    "Firebase ainda nao configurado.\n"
                    "Configure FIREBASE_API_KEY em "
                    "firebase_config.py antes de iniciar sessao."
                ),
                size_hint_y=None,
                height=dp(100),
            ))
            self.add_status(page)
            return

        if self.session_manager and self.session_manager.is_logged_in:
            self.show_profile(page)
        else:
            self.show_login(page)

    def show_login(self, page):
        page.add_widget(Label(
            text="Iniciar sessao ou criar conta",
            size_hint_y=None,
            height=dp(40),
        ))

        self.email_input = TextInput(
            hint_text="E-mail",
            multiline=False,
            size_hint_y=None,
            height=dp(48),
        )
        page.add_widget(self.email_input)

        self.password_input = TextInput(
            hint_text="Palavra-passe",
            password=True,
            multiline=False,
            size_hint_y=None,
            height=dp(48),
        )
        page.add_widget(self.password_input)

        self.name_input = TextInput(
            hint_text="Nome para registo",
            multiline=False,
            size_hint_y=None,
            height=dp(48),
        )
        page.add_widget(self.name_input)

        self.add_button(page, "Entrar", self.login)
        self.add_button(page, "Criar conta", self.register)
        self.add_button(page, "Recuperar palavra-passe",
                        self.reset_password)
        self.add_status(page)

    def _run_async(
        self, operation, on_success,
        busy_message="A processar...",
        error_prefix="Erro"
    ):
        if getattr(self, "_operation_busy", False):
            self.set_status("Aguarde: existe uma operacao em curso.")
            return

        self._operation_busy = True
        page_token = getattr(self, "_page_token", 0)
        self.set_status(busy_message)

        def worker():
            try:
                result = operation()
            except Exception as exc:
                Clock.schedule_once(
                    lambda dt, error=exc: self._finish_async(
                        page_token, None, error, on_success, error_prefix
                    ), 0
                )
            else:
                Clock.schedule_once(
                    lambda dt, value=result: self._finish_async(
                        page_token, value, None, on_success, error_prefix
                    ), 0
                )

        threading.Thread(target=worker, daemon=True).start()

    def _finish_async(
        self, page_token, result, error, on_success, error_prefix
    ):
        self._operation_busy = False

        if page_token != getattr(self, "_page_token", 0):
            return

        if error is not None:
            self.set_status(f"{error_prefix}: {error}")
            return

        try:
            on_success(result)
        except Exception as exc:
            self.set_status(f"Erro na interface: {exc}")

    def login(self, *_):
        email = self.email_input.text.strip()
        password = self.password_input.text

        def operation():
            session = self.auth.login(email, password)
            self.session_manager.start(session)
            return session

        def success(_):
            self.show_account()
            self.set_status("Sessao iniciada com sucesso.")

        self._run_async(
            operation, success,
            "A iniciar sessao...", "Falha no login"
        )

    def register(self, *_):
        email = self.email_input.text.strip()
        password = self.password_input.text
        name = self.name_input.text.strip()

        def operation():
            session = self.auth.register(email, password, name)
            self.session_manager.start(session)

            message = "Conta criada com sucesso."
            try:
                self.auth.send_email_verification(session["id_token"])
                message += " Verificacao enviada por e-mail."
            except Exception:
                message += " Nao foi possivel enviar a verificacao."

            return message

        def success(message):
            self.show_account()
            self.set_status(message)

        self._run_async(
            operation, success,
            "A criar conta...", "Falha no registo"
        )

    def reset_password(self, *_):
        email = self.email_input.text.strip()

        if not email:
            self.set_status("Introduza primeiro o seu e-mail.")
            return

        def operation():
            self.auth.send_password_reset(email)

        self._run_async(
            operation,
            lambda _: self.set_status(
                "Pedido de recuperacao enviado."
            ),
            "A enviar pedido...", "Erro na recuperacao"
        )

    def show_profile(self, page):
        loading = Label(
            text="A carregar perfil...",
            size_hint_y=None,
            height=dp(150),
            halign="left",
            valign="middle",
        )
        self.profile_photo_image = Image(
            source="",
            size_hint=(None, None),
            size=(dp(130), dp(130)),
            allow_stretch=True,
            keep_ratio=True,
        )
        page.add_widget(self.profile_photo_image)
        page.add_widget(loading)

        self.add_status(page)
        status_widget = self.status

        def operation():
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

            return profile, photo_path, photo_service

        def success(result):
            profile, photo_path, photo_service = result
            self.photo_service = photo_service

            if photo_path:
                self.profile_photo_image.source = photo_path
                self.profile_photo_image.reload()

            loading.text = (
                f"Nome: {profile.get('name') or 'Nao definido'}\n"
                f"E-mail: {profile.get('email')}\n"
                f"UID: {profile.get('uid')}\n"
                f"Metodo: {profile.get('provider')}\n"
                f"E-mail verificado: {profile.get('email_verified')}\n"
                f"Modelos locais: {len(self.alpha.list_models())}"
            )

            self._profile_label = loading

            self.profile_name_input = TextInput(
                hint_text="Nome do utilizador",
                text=profile.get("name") or "",
                multiline=False,
                size_hint_y=None,
                height=dp(48),
            )
            page.add_widget(self.profile_name_input)

            self.add_button(
                page,
                "Alterar fotografia",
                self._open_profile_gallery,
            )

            self.add_button(
                page, "Guardar nome",
                self.update_profile
            )

            self.add_button(
                page, "Enviar verificacao de e-mail",
                self.send_verification
            )
            self.add_button(
                page, "Sincronizar modelos",
                self.sync_models
            )
            self.add_button(
                page, "Terminar sessao",
                self.logout
            )
            self.set_status("Perfil carregado.")

        self._run_async(
            operation, success,
            "A carregar perfil...",
            "Nao foi possivel carregar o perfil"
        )


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

    def update_profile(self, *_):
        name = self.profile_name_input.text.strip()

        if not name:
            self.set_status("Introduza um nome.")
            return

        if len(name) > 80:
            self.set_status("O nome deve ter no maximo 80 caracteres.")
            return

        def operation():
            session = self.session_manager.ensure_valid()
            profile = UserProfile(self.auth, session)
            data = profile.update(name=name)
            self.session_manager.update(session)
            return data

        def success(data):
            self.profile_name_input.text = data.get("name") or name

            self._profile_label.text = (
                f"Nome: {data.get('name') or 'Nao definido'}\\n"
                f"E-mail: {data.get('email')}\\n"
                f"UID: {data.get('uid')}\\n"
                f"Metodo: {data.get('provider')}\\n"
                f"E-mail verificado: {data.get('email_verified')}\\n"
                f"Modelos locais: {len(self.alpha.list_models())}"
            )

            self.set_status("Perfil atualizado.")

        self._run_async(
            operation,
            success,
            "A atualizar perfil...",
            "Erro ao atualizar perfil"
        )

    def send_verification(self, *_):
        def operation():
            token = self.session_manager.get_id_token()
            self.auth.send_email_verification(token)

        self._run_async(
            operation,
            lambda _: self.set_status("Verificacao enviada."),
            "A enviar verificacao...", "Erro no envio"
        )

    def sync_models(self, *_):
        def operation():
            self.session_manager.ensure_valid()
            sync = FirebaseSync(
                self.session_manager.session,
                session_manager=self.session_manager,
            )
            return sync.sync_models(self.alpha)

        def success(result):
            self.set_status(
                f"Sincronizacao concluida.\n"
                f"Enviados: {len(result['uploaded'])}\n"
                f"Recebidos: {len(result['downloaded'])}"
            )

        self._run_async(
            operation, success,
            "A sincronizar modelos...",
            "Erro na sincronizacao"
        )

    def logout(self, *_):
        if getattr(self, "_operation_busy", False):
            self.set_status("Aguarde a operacao em curso.")
            return

        self.session_manager.logout()
        self.show_account()
        self.set_status(
            "Sessao terminada. Treino local continua disponivel."
        )


if __name__ == "__main__":
    AlphaTrainerAndroid().run()
