"""
Alpha Core V1.2
Núcleo central de integração do AlphaTrainer.
"""

from models.ai_factory import AIModelFactory
from registry.registry import ModelRegistry
from runtime import AlphaRuntime
from training.trainer import Trainer
from evaluation.evaluator import evaluate_model
from evaluation.report import generate_report


class AlphaCore:

    def __init__(self):
        self.registry = ModelRegistry()
        self.factory = AIModelFactory()
        self.runtime = AlphaRuntime()

    @staticmethod
    def _validate_name(name):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Nome do modelo invalido.")

    def create_model(self, name, layer_sizes, activation="relu"):
        return self.create_model_by_type(
            name,
            "feedforward",
            layer_sizes=layer_sizes,
            activation=activation,
        )

    def create_model_by_type(self, name, model_type, **config):
        """Create and register a model using any registered factory builder."""
        self._validate_name(name)
        model = self.factory.create(model_type, **config)
        return self.registry.register(name, model)

    def create_language_model(self, name, vocabulary, **config):
        return self.create_model_by_type(
            name, "language_model", vocabulary=vocabulary, **config
        )

    def register_model(self, name, model):
        return self.registry.register(name, model)

    def get_model(self, name):
        return self.registry.get(name)

    def list_models(self):
        return self.registry.list_models()

    def remove_model(self, name):
        return self.registry.remove(name)

    def run_model(self, name, inputs):
        return self.runtime.run(self.get_model(name), inputs)

    def predict_next(self, name, token_ids):
        return self.runtime.predict_next(self.get_model(name), token_ids)

    def generate_tokens(self, name, token_ids, max_new_tokens=20, eos_id=None):
        return self.runtime.generate(
            self.get_model(name),
            token_ids,
            max_new_tokens=max_new_tokens,
            eos_id=eos_id,
        )

    def generate_text(self, name, text, max_new_tokens=20):
        return self.runtime.generate_text(
            self.get_model(name), text, max_new_tokens=max_new_tokens
        )

    def train_model(self, name, dataset, optimizer, epochs=100):
        model = self.get_model(name)
        trainer = Trainer(optimizer)
        return trainer.fit(model, dataset, epochs=epochs)

    def evaluate(self, name, dataset):
        return evaluate_model(self.get_model(name), dataset)

    def report(self, name, dataset):
        return generate_report(self.get_model(name), dataset)

    def model_count(self):
        return self.registry.count()
