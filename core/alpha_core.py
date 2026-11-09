"""
Alpha Core V1.1
Nucleo central de integracao do AlphaTrainer.
"""

from functools import wraps
from threading import RLock

def _synchronized(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper

from models.factory import create_model
from registry.registry import ModelRegistry
from training.trainer import Trainer
from evaluation.evaluator import evaluate_model
from evaluation.report import generate_report
from evaluation.classification import classification_report
from checkpoints import save_any, load_any
from neural.network import Network
from transformer.language_model import LanguageModel


class AlphaCore:

    def __init__(self):
        self._lock = RLock()
        self.registry = ModelRegistry()

    # ------------------------------------------------------------------
    # Gestao de modelos

    @_synchronized
    def create_model(
        self,
        name,
        layer_sizes,
        activation="relu",
        output_activation=None,
        init="auto",
        seed=42
    ):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Nome do modelo invalido.")

        if name in self.registry.models:
            raise ValueError("Ja existe um modelo com esse nome")

        model = create_model(
            layer_sizes,
            activation=activation,
            output_activation=output_activation,
            init=init,
            seed=seed
        )

        return self.registry.register(name, model)

    @_synchronized
    def create_language_model(self, name, texts, **options):
        """Cria um modelo de linguagem com vocabulario construido de texts."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Nome do modelo invalido.")

        if name in self.registry.models:
            raise ValueError("Ja existe um modelo com esse nome")

        model = LanguageModel.from_texts(texts, **options)

        return self.registry.register(name, model)

    @_synchronized
    def register_model(self, name, model):
        return self.registry.register(name, model)

    @_synchronized
    def get_model(self, name):
        return self.registry.get(name)

    @_synchronized
    def list_models(self):
        return self.registry.list_models()

    @_synchronized
    def remove_model(self, name):
        return self.registry.remove(name)

    @_synchronized
    def model_count(self):
        return self.registry.count()

    @_synchronized
    def describe_model(self, name):
        """Resumo legivel de um modelo registado."""
        model = self.get_model(name)

        if isinstance(model, Network):
            return {
                "name": name,
                "type": "rede neuronal",
                "layers": model.layer_sizes(),
                "activations": [
                    layer.neurons[0].activation_name
                    for layer in model.layers
                ],
                "parameters": model.parameter_count(),
            }

        if isinstance(model, LanguageModel):
            return {
                "name": name,
                "type": "modelo de linguagem",
                "vocabulary": model.vocabulary.size,
                "embedding_dim": model.embedding_dim,
                "hidden_dim": model.hidden_dim,
                "num_layers": model.num_layers,
                "max_length": model.max_length,
                "parameters": model.parameter_count(),
            }

        return {"name": name, "type": type(model).__name__}

    # ------------------------------------------------------------------
    # Treino / avaliacao

    @_synchronized
    def train_model(
        self,
        name,
        dataset,
        optimizer,
        epochs=100,
        **options
    ):
        model = self.get_model(name)

        trainer = Trainer(optimizer)

        return trainer.fit(
            model,
            dataset,
            epochs=epochs,
            **options
        )

    @_synchronized
    def train_language_model(self, name, texts, epochs=50, **options):
        model = self.get_model(name)

        if not isinstance(model, LanguageModel):
            raise TypeError("O modelo nao e um modelo de linguagem.")

        return model.fit(texts, epochs=epochs, **options)

    @_synchronized
    def evaluate(self, name, dataset):
        model = self.get_model(name)

        return evaluate_model(model, dataset)

    @_synchronized
    def report(self, name, dataset):
        model = self.get_model(name)

        return generate_report(model, dataset)

    @_synchronized
    def evaluate_classification(self, name, dataset, classes=None):
        """Acuracia, matriz de confusao e metricas por classe."""
        model = self.get_model(name)

        return classification_report(model, dataset, classes)

    @_synchronized
    def predict_class(self, name, inputs):
        model = self.get_model(name)

        if not isinstance(model, Network):
            raise TypeError("O modelo nao e uma rede neuronal.")

        return model.predict_class(list(inputs))

    @_synchronized
    def predict(self, name, inputs):
        model = self.get_model(name)

        if not isinstance(model, Network):
            raise TypeError("O modelo nao e uma rede neuronal.")

        return model.forward(list(inputs))

    @_synchronized
    def generate(self, name, prompt="", **options):
        model = self.get_model(name)

        if not isinstance(model, LanguageModel):
            raise TypeError("O modelo nao e um modelo de linguagem.")

        return model.generate(prompt, **options)

    # ------------------------------------------------------------------
    # Persistencia

    @_synchronized
    def save_model(self, name, path, metadata=None):
        save_any(self.get_model(name), path, metadata=metadata)
        return path

    @_synchronized
    def load_model(self, name, path):
        """Carrega um checkpoint e regista-o com o nome indicado."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Nome do modelo invalido.")

        if name in self.registry.models:
            raise ValueError("Ja existe um modelo com esse nome")

        return self.registry.register(name, load_any(path))
