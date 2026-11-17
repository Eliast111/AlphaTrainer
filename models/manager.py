"""Gestao de ciclo de vida e metadados persistiveis dos modelos."""

import json
from copy import deepcopy
from dataclasses import asdict

from models.contracts import ModelMetadata, ModelCapabilities
from models.adapters import DenseNetworkAdapter, TransformerLanguageAdapter
from neural.network import Network
from transformer.language_model import LanguageModel


METADATA_FORMAT = "alpha_model_metadata_v1"
TRAINING_STATES = {
    "created", "queued", "training", "completed", "failed", "cancelled"
}


class ModelManager:
    def __init__(self, registry):
        self.registry = registry
        self._metadata = {}
        self._adapters = {}
        self._adapter_catalog = {}

    @staticmethod
    def _infer(model):
        if isinstance(model, Network):
            return DenseNetworkAdapter(), {
                "layers": model.layer_sizes(),
                "parameters": model.parameter_count(),
            }
        if isinstance(model, LanguageModel):
            return TransformerLanguageAdapter(), {
                "vocabulary_size": model.vocabulary.size,
                "embedding_dim": model.embedding_dim,
                "hidden_dim": model.hidden_dim,
                "num_layers": model.num_layers,
                "max_length": model.max_length,
                "parameters": model.parameter_count(),
            }

        class CustomAdapter:
            model_type = "custom"
            architecture = type(model).__module__ + "." + type(model).__name__
            capabilities = ModelCapabilities()

            @staticmethod
            def validate(candidate):
                if candidate is None:
                    raise ValueError("Modelo invalido.")

        return CustomAdapter(), {"parameters": None}

    def register(
        self, name, model, config=None, dataset_id=None, adapter=None
    ):
        if adapter is None:
            adapter, inferred = self._infer(model)
        else:
            required = ("model_type", "architecture", "capabilities", "validate")
            missing = [
                field for field in required
                if not hasattr(adapter, field)
            ]
            if missing:
                raise TypeError(
                    "Adaptador incompleto: " + ", ".join(missing)
                )

            if not isinstance(adapter.model_type, str) or not adapter.model_type.strip():
                raise ValueError("Tipo de modelo invalido.")

            if not isinstance(adapter.architecture, str) or not adapter.architecture.strip():
                raise ValueError("Arquitetura invalida.")

            if not isinstance(adapter.capabilities, ModelCapabilities):
                raise TypeError("Capacidades devem ser ModelCapabilities.")

            if not callable(adapter.validate):
                raise TypeError("O adaptador deve implementar validate().")

            inferred = {"parameters": None}

        adapter.validate(model)
        self.registry.register(name, model)
        metadata = ModelMetadata(
            name=name,
            model_type=adapter.model_type,
            architecture=adapter.architecture,
            config=dict(inferred, **(config or {})),
            dataset_id=dataset_id,
        )
        self._metadata[name] = metadata

        if not isinstance(model, (Network, LanguageModel)):
            self.register_adapter(adapter)

        if adapter is not None and not isinstance(
            model, (Network, LanguageModel)
        ):
            self._adapters[name] = adapter

        return model

    def register_adapter(self, adapter):
        """Regista explicitamente um adaptador confiavel em memoria."""
        required = ("model_type", "architecture", "capabilities", "validate")
        for field in required:
            if not hasattr(adapter, field):
                raise TypeError("Adaptador incompleto: " + field)

        if not isinstance(adapter.model_type, str) or not adapter.model_type.strip():
            raise ValueError("Tipo de modelo invalido.")
        if not isinstance(adapter.architecture, str) or not adapter.architecture.strip():
            raise ValueError("Arquitetura invalida.")
        if not isinstance(adapter.capabilities, ModelCapabilities):
            raise TypeError("Capacidades devem ser ModelCapabilities.")
        if not callable(adapter.validate):
            raise TypeError("O adaptador deve implementar validate().")

        key = (adapter.model_type, adapter.architecture)
        existing = self._adapter_catalog.get(key)

        if existing is not None and existing is not adapter:
            raise ValueError("Ja existe um adaptador registado para esta arquitetura.")

        self._adapter_catalog[key] = adapter
        return adapter

    def export_custom_state(self, name):
        """Exporta o estado de um modelo personalizado como JSON seguro."""
        if name not in self.registry.models:
            raise KeyError("Modelo nao encontrado: " + name)

        model = self.registry.get(name)
        if isinstance(model, (Network, LanguageModel)):
            raise ValueError("Use o checkpoint nativo para este modelo.")

        adapter = self.adapter_for(name)
        exporter = getattr(adapter, "export_state", None)
        restorer = getattr(adapter, "restore_state", None)

        if not callable(exporter) or not callable(restorer):
            raise TypeError("Adaptador sem suporte de persistencia.")

        state = exporter(model)
        try:
            state = json.loads(json.dumps(state, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("Estado do modelo nao e JSON seguro.") from error

        return {
            "format": "alpha_custom_model_state_v1",
            "model_type": adapter.model_type,
            "architecture": adapter.architecture,
            "state": state,
        }

    def restore_custom_model(self, name, payload):
        """Restaura usando exclusivamente um adaptador previamente registado."""
        if not isinstance(payload, dict):
            raise TypeError("Estado personalizado invalido.")

        if payload.get("format") != "alpha_custom_model_state_v1":
            raise ValueError("Formato de modelo personalizado desconhecido.")

        model_type = payload.get("model_type")
        architecture = payload.get("architecture")

        if not isinstance(model_type, str) or not isinstance(architecture, str):
            raise ValueError("Identidade do adaptador invalida.")

        adapter = self._adapter_catalog.get((model_type, architecture))
        if adapter is None:
            raise KeyError("Adaptador confiavel nao registado.")

        restorer = getattr(adapter, "restore_state", None)
        if not callable(restorer):
            raise TypeError("Adaptador sem suporte de recuperacao.")

        try:
            state = json.loads(json.dumps(payload.get("state"), allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("Estado guardado nao e JSON seguro.") from error

        model = restorer(state)
        adapter.validate(model)
        return self.register(name, model, adapter=adapter)

    def adapter_for(self, name):
        """Devolve o adaptador associado ao modelo."""
        if name not in self.registry.models:
            raise KeyError("Modelo nao encontrado: " + name)

        if name in self._adapters:
            return self._adapters[name]

        adapter, _ = self._infer(self.registry.get(name))
        return adapter

    def capabilities(self, name):
        return self.adapter_for(name).capabilities

    def metadata(self, name):
        if name not in self.registry.models:
            raise KeyError("Modelo nao encontrado: " + name)
        if name not in self._metadata:
            model = self.registry.get(name)
            adapter = self.adapter_for(name)
            _, inferred = self._infer(model)
            self._metadata[name] = ModelMetadata(
                name=name,
                model_type=adapter.model_type,
                architecture=adapter.architecture,
                config=inferred,
            )
        return self._metadata[name]

    def list_metadata(self):
        return [self.metadata(name) for name in self.registry.list_models()]

    def update_training(self, name, status=None, metrics=None, dataset_id=None):
        item = self.metadata(name)
        if status is not None:
            if status not in TRAINING_STATES:
                raise ValueError("Estado de treino invalido.")
            item.training_status = status
        if metrics is not None:
            item.metrics.append(deepcopy(metrics))
        if dataset_id is not None:
            item.dataset_id = dataset_id
        item.touch()
        return item

    def export_metadata(self, name):
        """Exporta metadados JSON seguros, incluindo capacidades inferidas."""
        item = self.metadata(name)
        payload = asdict(item)
        payload["metadata_format"] = METADATA_FORMAT
        payload["capabilities"] = asdict(self.capabilities(name))

        # Impede que objetos arbitrarios ou NaN entrem no manifesto.
        json.dumps(payload, allow_nan=False)
        return payload

    def restore_metadata(self, name, payload):
        """Restaura metadados validados para um modelo ja registado."""
        if name not in self.registry.models:
            raise KeyError("Modelo nao encontrado: " + name)
        if not isinstance(payload, dict):
            raise TypeError("Metadados devem ser um dicionario.")
        if payload.get("metadata_format") != METADATA_FORMAT:
            raise ValueError("Formato de metadados desconhecido.")

        model = self.registry.get(name)
        adapter = self.adapter_for(name)
        _, inferred = self._infer(model)
        adapter.validate(model)

        if payload.get("name") != name:
            raise ValueError("Nome dos metadados nao corresponde ao modelo.")
        if payload.get("model_type") != adapter.model_type:
            raise ValueError("Tipo de modelo dos metadados incompativel.")
        if payload.get("architecture") != adapter.architecture:
            raise ValueError("Arquitetura dos metadados incompativel.")

        model_id = payload.get("model_id")
        config = payload.get("config", inferred)
        dataset_id = payload.get("dataset_id")
        status = payload.get("training_status", "created")
        metrics = payload.get("metrics", [])
        checkpoints = payload.get("checkpoints", [])
        version = payload.get("version", 1)
        created_at = payload.get("created_at")
        updated_at = payload.get("updated_at")

        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("Identidade do modelo invalida.")
        if not isinstance(config, dict):
            raise ValueError("Configuracao dos metadados invalida.")
        if dataset_id is not None and not isinstance(dataset_id, str):
            raise ValueError("Identificador do dataset invalido.")
        if status not in TRAINING_STATES:
            raise ValueError("Estado de treino invalido.")
        if not isinstance(metrics, list) or not isinstance(checkpoints, list):
            raise ValueError("Historico de treino invalido.")
        if (
            not isinstance(version, int)
            or isinstance(version, bool)
            or version < 1
        ):
            raise ValueError("Versao do modelo invalida.")
        if not isinstance(created_at, str) or not isinstance(updated_at, str):
            raise ValueError("Datas dos metadados invalidas.")

        expected_capabilities = asdict(adapter.capabilities)
        saved_capabilities = payload.get("capabilities")

        # Compatibilidade com manifestos anteriores a Fase 5, que ainda nao
        # registavam as capacidades forward e resume.
        legacy_capabilities = {
            key: value for key, value in expected_capabilities.items()
            if key not in {"forward", "resume"}
        }
        if saved_capabilities not in (
            expected_capabilities, legacy_capabilities
        ):
            raise ValueError("Capacidades guardadas nao correspondem a arquitetura.")

        # Confirma que todos os campos persistidos sao dados JSON finitos.
        json.dumps(
            {
                "config": config,
                "metrics": metrics,
                "checkpoints": checkpoints,
            },
            allow_nan=False,
        )

        restored = ModelMetadata(
            name=name,
            model_type=adapter.model_type,
            architecture=adapter.architecture,
            model_id=model_id,
            config=deepcopy(config),
            dataset_id=dataset_id,
            training_status=status,
            metrics=deepcopy(metrics),
            checkpoints=deepcopy(checkpoints),
            version=version,
            created_at=created_at,
            updated_at=updated_at,
        )
        self._metadata[name] = restored
        return restored

    def rename(self, old_name, new_name):
        if old_name not in self.registry.models:
            raise KeyError("Modelo nao encontrado: " + old_name)
        if not isinstance(new_name, str) or not new_name.strip():
            raise ValueError("Nome do modelo invalido.")
        if new_name in self.registry.models:
            raise ValueError("Ja existe um modelo com esse nome")
        model = self.registry.models.pop(old_name)
        self.registry.models[new_name] = model

        adapter = self._adapters.pop(old_name, None)
        if adapter is not None:
            self._adapters[new_name] = adapter

        item = self._metadata.pop(old_name, None)
        if item is not None:
            item.name = new_name
            item.touch()
            self._metadata[new_name] = item
        return model

    def duplicate(self, name, new_name):
        if name not in self.registry.models:
            raise KeyError("Modelo nao encontrado: " + name)
        adapter = self._adapters.get(name)
        return self.register(
            new_name,
            deepcopy(self.registry.get(name)),
            adapter=adapter,
        )

    def remove(self, name):
        model = self.registry.remove(name)
        self._metadata.pop(name, None)
        self._adapters.pop(name, None)
        return model
