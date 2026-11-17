"""Interface Python local para utilizar o AlphaTrainer sem servicos externos.

O SDK recebe uma instancia AlphaCore; nao cria servidores, nao faz pedidos
HTTP e nao envia datasets para terceiros.
"""

class LocalSDKError(RuntimeError):
    """Erro de utilizacao da interface SDK local."""


class LocalLearningSDK:
    """Facade estavel e pequena sobre as operacoes publicas de AlphaCore."""

    def __init__(self, core):
        required = (
            "create_model", "register_model", "list_models",
            "model_metadata", "model_capabilities", "train_universal",
            "runtime_predict", "runtime_forward", "evaluate_universal",
        )
        missing = [name for name in required if not callable(getattr(core, name, None))]
        if missing:
            raise TypeError("AlphaCore incompleto: " + ", ".join(missing))
        self._core = core

    def create_dense_model(self, name, layer_sizes, **options):
        return self._core.create_model(name, layer_sizes, **options)

    def create_language_model(self, name, texts, **options):
        method = getattr(self._core, "create_language_model", None)
        if not callable(method):
            raise LocalSDKError("Criacao de modelos de linguagem indisponivel.")
        return method(name, texts, **options)

    def register_model(self, name, model, *, adapter=None):
        if adapter is None:
            return self._core.register_model(name, model)
        return self._core.register_model(name, model, adapter=adapter)

    def list_models(self):
        return self._core.list_models()

    def metadata(self, name):
        return self._core.model_metadata(name)

    def capabilities(self, name):
        return self._core.model_capabilities(name)

    def train(self, name, dataset, *, optimizer=None, epochs=100, **options):
        capabilities = self.capabilities(name)
        if not capabilities.get("train", False):
            raise LocalSDKError("Este modelo nao declara suporte a treino.")

        requests_session = (
            bool(options.get("resume"))
            or options.get("checkpoint_path") is not None
        )
        if requests_session and not capabilities.get("resume", False):
            raise LocalSDKError(
                "Este modelo nao suporta checkpoints de sessao/retoma local."
            )

        return self._core.train_universal(
            name, dataset, optimizer=optimizer, epochs=epochs, **options
        )

    def predict(self, name, inputs):
        if not self.capabilities(name).get("predict", False):
            raise LocalSDKError("Este modelo nao suporta previsao.")
        return self._core.runtime_predict(name, inputs)

    def forward(self, name, inputs):
        if not self.capabilities(name).get("forward", False):
            raise LocalSDKError("Este modelo nao suporta forward.")
        return self._core.runtime_forward(name, inputs)

    def generate(self, name, prompt="", **options):
        if not self.capabilities(name).get("generate", False):
            raise LocalSDKError("Este modelo nao suporta geracao de texto.")
        method = getattr(self._core, "runtime_generate", None)
        if not callable(method):
            raise LocalSDKError("Runtime de geracao indisponivel.")
        return method(name, prompt, **options)

    def evaluate(self, name, dataset):
        if not self.capabilities(name).get("evaluate", False):
            raise LocalSDKError("Este modelo nao suporta avaliacao.")
        return self._core.evaluate_universal(name, dataset)
