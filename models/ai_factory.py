"""Extensible, dependency-free factory for AlphaTrainer models."""

from models.factory import create_model as create_feedforward_model


class AIModelFactory:
    """Create built-in or registered model types through one interface."""

    def __init__(self):
        self._builders = {
            "feedforward": self._build_feedforward,
            "language_model": self._build_language_model,
        }

    @property
    def model_types(self):
        """Return supported model type names in registration order."""
        return tuple(self._builders)

    def register(self, model_type, builder):
        if not isinstance(model_type, str) or not model_type.strip():
            raise ValueError("O tipo do modelo deve ser uma string não vazia.")
        if not callable(builder):
            raise TypeError("O builder deve ser chamável.")
        if model_type in self._builders:
            raise ValueError("Já existe um builder para esse tipo de modelo.")
        self._builders[model_type] = builder
        return builder

    def create(self, model_type, **config):
        if not isinstance(model_type, str) or not model_type.strip():
            raise ValueError("O tipo do modelo deve ser uma string não vazia.")
        try:
            builder = self._builders[model_type]
        except KeyError as error:
            raise ValueError(f"Tipo de modelo não suportado: {model_type}") from error
        return builder(**config)

    @staticmethod
    def _build_feedforward(layer_sizes, activation="relu"):
        return create_feedforward_model(layer_sizes, activation=activation)

    @staticmethod
    def _build_language_model(
        vocabulary,
        embedding_dim=16,
        hidden_dim=32,
        num_layers=1,
        max_length=128,
        seed=42,
    ):
        from transformer.language_model import LanguageModel

        return LanguageModel(
            vocabulary=vocabulary,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_length=max_length,
            seed=seed,
        )


__all__ = ["AIModelFactory"]
