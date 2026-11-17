"""Contratos e metadados comuns para os modelos AlphaTrainer.

Os contratos descrevem capacidades sem obrigar modalidades diferentes a
implementar operações que nao fazem sentido para elas.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from uuid import uuid4


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ModelMetadata:
    name: str
    model_type: str
    architecture: str
    model_id: str = field(default_factory=lambda: str(uuid4()))
    config: Dict[str, Any] = field(default_factory=dict)
    dataset_id: Optional[str] = None
    training_status: str = "created"
    metrics: list = field(default_factory=list)
    checkpoints: list = field(default_factory=list)
    version: int = 1
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def touch(self):
        self.updated_at = utc_now()


@dataclass(frozen=True)
class ModelCapabilities:
    train: bool = False
    evaluate: bool = False
    predict: bool = False
    generate: bool = False
    forward: bool = False
    resume: bool = False
    export: bool = False


@runtime_checkable
class ModelAdapter(Protocol):
    """Interface minima: identidade, capacidades e validacao."""

    @property
    def model_type(self) -> str: ...

    @property
    def architecture(self) -> str: ...

    @property
    def capabilities(self) -> ModelCapabilities: ...

    def validate(self, model: Any) -> None: ...


@runtime_checkable
class PredictiveModel(Protocol):
    def forward(self, inputs): ...


@runtime_checkable
class GenerativeModel(Protocol):
    def generate(self, prompt="", **options): ...


@runtime_checkable
class ParameterizedModel(Protocol):
    def parameters(self): ...
