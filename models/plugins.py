"""Registo de extensoes em processo; nunca importa codigo a partir de ficheiros."""


class PluginRegistry:
    def __init__(self):
        self._plugins = {}

    def register(self, name, builder):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Nome do plugin invalido.")
        if not callable(builder):
            raise TypeError("O construtor do plugin deve ser callable.")
        if name in self._plugins:
            raise ValueError("Plugin ja registado: " + name)
        self._plugins[name] = builder

    def create(self, name, **config):
        if name not in self._plugins:
            raise KeyError("Plugin desconhecido: " + name)
        return self._plugins[name](**config)

    def list_plugins(self):
        return sorted(self._plugins)

    def unregister(self, name):
        if name not in self._plugins:
            raise KeyError("Plugin desconhecido: " + name)
        return self._plugins.pop(name)
