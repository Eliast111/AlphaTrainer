"""
Alpha Core V1.0
Nucleo central de integracao do AlphaTrainer.
"""

from models.factory import create_model
from registry.registry import ModelRegistry
from training.trainer import Trainer
from evaluation.evaluator import evaluate_model
from evaluation.report import generate_report


class AlphaCore:

    def __init__(self):
        self.registry = ModelRegistry()

    def create_model(self, name, layer_sizes, activation="relu"):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Nome do modelo invalido.")

        model = create_model(
            layer_sizes,
            activation=activation
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

    def train_model(
        self,
        name,
        dataset,
        optimizer,
        epochs=100
    ):
        model = self.get_model(name)

        trainer = Trainer(optimizer)

        return trainer.fit(
            model,
            dataset,
            epochs=epochs
        )

    def evaluate(self, name, dataset):
        model = self.get_model(name)

        return evaluate_model(model, dataset)

    def report(self, name, dataset):
        model = self.get_model(name)

        return generate_report(model, dataset)

    def model_count(self):
        return self.registry.count()
