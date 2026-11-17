"""
Alpha Core V1.1
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

    def create_model(self, name, layer_sizes, activation="relu"):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Nome do modelo invalido.")

        model = self.factory.create(
            "feedforward", layer_sizes=layer_sizes, activation=activation
        )

        return self.registry.register(name, model)

    def create_language_model(self, name, vocabulary, **config):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Nome do modelo invalido.")
        model = self.factory.create(
            "language_model", vocabulary=vocabulary, **config
        )
        return self.registry.register(name, model)

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
