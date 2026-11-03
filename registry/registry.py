"""
Alpha Model Registry V0.1
Gestao e organizacao de modelos.
"""


class ModelRegistry:

    def __init__(self):
        self.models = {}

    def register(self, name, model):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Nome do modelo invalido")

        if name in self.models:
            raise ValueError("Ja existe um modelo com esse nome")

        if model is None:
            raise ValueError("Modelo invalido")

        self.models[name] = model
        return model

    def get(self, name):
        if name not in self.models:
            raise KeyError("Modelo nao encontrado: " + name)

        return self.models[name]

    def list_models(self):
        return list(self.models.keys())

    def remove(self, name):
        if name not in self.models:
            raise KeyError("Modelo nao encontrado: " + name)

        return self.models.pop(name)

    def count(self):
        return len(self.models)
