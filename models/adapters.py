"""Adaptadores para as arquiteturas que ja existem no AlphaTrainer."""

from models.contracts import ModelCapabilities
from neural.network import Network
from transformer.language_model import LanguageModel


class DenseNetworkAdapter:
    model_type = "numeric"
    architecture = "dense"
    capabilities = ModelCapabilities(
        train=True, evaluate=True, predict=True, forward=True,
        resume=True, export=True
    )

    def validate(self, model):
        if not isinstance(model, Network):
            raise TypeError("Era esperada uma rede neuronal densa.")


class TransformerLanguageAdapter:
    model_type = "text"
    architecture = "transformer_decoder"
    capabilities = ModelCapabilities(
        train=True, evaluate=True, predict=True, generate=True,
        forward=True, export=True
    )

    def validate(self, model):
        if not isinstance(model, LanguageModel):
            raise TypeError("Era esperado um modelo Transformer de linguagem.")
