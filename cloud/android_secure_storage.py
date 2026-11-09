"""Armazenamento cifrado de sessoes Firebase no Android."""

import base64
import json


class AndroidKeystoreSessionStorage:
    ALIAS = "AlphaTrainerSessionKey"
    PREFS = "alphatrainer_secure"
    PREF_KEY = "encrypted_session"

    def __init__(self, context=None):
        self.context = context

    def _android(self):
        from jnius import autoclass, jarray

        PythonActivity = autoclass(
            "org.kivy.android.PythonActivity"
        )
        context = self.context or PythonActivity.mActivity

        if context is None:
            raise RuntimeError("Contexto Android indisponivel.")

        KeyStore = autoclass("java.security.KeyStore")
        KeyGenerator = autoclass("javax.crypto.KeyGenerator")
        KeyProperties = autoclass(
            "android.security.keystore.KeyProperties"
        )
        Builder = autoclass(
            "android.security.keystore.KeyGenParameterSpec$Builder"
        )
        Cipher = autoclass("javax.crypto.Cipher")
        GCMParameterSpec = autoclass(
            "javax.crypto.spec.GCMParameterSpec"
        )

        return (
            context, jarray, KeyStore, KeyGenerator,
            KeyProperties, Builder, Cipher, GCMParameterSpec
        )

    @staticmethod
    def _java_bytes(jarray, data):
        signed = [value if value < 128 else value - 256
                  for value in data]
        return jarray("b")(signed)

    @staticmethod
    def _python_bytes(data):
        return bytes((int(value) & 255) for value in data)

    def _get_key(self, create=True):
        (
            _, jarray, KeyStore, KeyGenerator,
            KeyProperties, Builder, _, _
        ) = self._android()

        store = KeyStore.getInstance("AndroidKeyStore")
        store.load(None)

        key = store.getKey(self.ALIAS, None)
        if key is not None:
            return key

        if not create:
            return None

        purposes = (
            KeyProperties.PURPOSE_ENCRYPT
            | KeyProperties.PURPOSE_DECRYPT
        )

        builder = Builder(self.ALIAS, purposes)
        builder.setBlockModes(
            jarray("java.lang.String")(["GCM"])
        )
        builder.setEncryptionPaddings(
            jarray("java.lang.String")(["NoPadding"])
        )
        builder.setRandomizedEncryptionRequired(True)

        generator = KeyGenerator.getInstance(
            "AES", "AndroidKeyStore"
        )
        generator.init(builder.build())
        return generator.generateKey()

    def _preferences(self):
        context = self._android()[0]
        return context.getSharedPreferences(self.PREFS, 0)

    def save(self, session):
        (
            _, jarray, _, _, _, _, Cipher, _
        ) = self._android()

        key = self._get_key()
        cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, key)

        plain = json.dumps(
            session, separators=(",", ":")
        ).encode("utf-8")

        encrypted = cipher.doFinal(
            self._java_bytes(jarray, plain)
        )

        iv = self._python_bytes(cipher.getIV())
        ciphertext = self._python_bytes(encrypted)

        envelope = json.dumps({
            "iv": base64.b64encode(iv).decode("ascii"),
            "data": base64.b64encode(ciphertext).decode("ascii"),
        })

        self._preferences().edit().putString(
            self.PREF_KEY, envelope
        ).apply()

    def load(self):
        (
            _, jarray, _, _, _, _, Cipher, GCMParameterSpec
        ) = self._android()

        saved = self._preferences().getString(
            self.PREF_KEY, None
        )

        if not saved:
            return None

        envelope = json.loads(saved)
        iv = base64.b64decode(envelope["iv"])
        ciphertext = base64.b64decode(envelope["data"])

        if len(iv) != 12:
            raise RuntimeError("IV da sessao invalido.")

        key = self._get_key(create=False)
        if key is None:
            raise RuntimeError("Chave Android da sessao inexistente.")

        cipher = Cipher.getInstance("AES/GCM/NoPadding")
        spec = GCMParameterSpec(
            128, self._java_bytes(jarray, iv)
        )
        cipher.init(Cipher.DECRYPT_MODE, key, spec)

        plain = cipher.doFinal(
            self._java_bytes(jarray, ciphertext)
        )

        return json.loads(
            self._python_bytes(plain).decode("utf-8")
        )

    def clear(self):
        self._preferences().edit().remove(
            self.PREF_KEY
        ).apply()
