"""Execucao universal de modelos atraves do ModelManager."""


class AIRuntimeError(RuntimeError):
    """Erro durante a execucao de um modelo."""


class AIRuntime:
    """Encaminha operacoes para os motores de IA registados."""

    def __init__(self, manager):
        self.manager = manager

    def _resolve(self, name):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Nome do modelo invalido.")

        model = self.manager.registry.get(name)
        adapter = self.manager.adapter_for(name)

        return model, adapter.capabilities

    def execute(self, name, operation, *args, **kwargs):
        model, capabilities = self._resolve(name)

        if operation == "predict":
            if not capabilities.predict:
                raise AIRuntimeError(
                    "Este modelo nao suporta previsao."
                )

            method = getattr(model, "predict", None)

            if not callable(method):
                method = getattr(model, "predict_next", None)

        elif operation == "generate":
            if not capabilities.generate:
                raise AIRuntimeError(
                    "Este modelo nao suporta geracao."
                )

            method = getattr(model, "generate", None)

        elif operation == "forward":
            if not getattr(capabilities, "forward", True):
                raise AIRuntimeError(
                    "Este modelo nao suporta propagacao direta (forward)."
                )
            method = getattr(model, "forward", None)

        else:
            raise AIRuntimeError(
                "Operacao desconhecida: " + str(operation)
            )

        if not callable(method):
            raise AIRuntimeError(
                "O modelo nao implementa a operacao solicitada."
            )

        return method(*args, **kwargs)

    def predict(self, name, inputs):
        return self.execute(name, "predict", inputs)

    def generate(self, name, prompt="", **options):
        return self.execute(
            name, "generate", prompt, **options
        )

    def forward(self, name, inputs):
        return self.execute(name, "forward", inputs)
