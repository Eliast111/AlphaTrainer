"""
Alpha Persistence Engine V2.0
Persistencia versionada de modelos, metadados e datasets.

Mantem compatibilidade de leitura com manifestos alpha_persistence_v1.
"""

import base64
import json
import uuid
from pathlib import Path

from checkpoints import load_any, save_any
from checkpoints.checkpoint import write_json_atomic
from data.dataset import Dataset
from neural.network import Network
from transformer.language_model import LanguageModel


MANIFEST_V1 = "alpha_persistence_v1"
MANIFEST_V2 = "alpha_persistence_v2"
MANIFEST_VERSION = 2


class PersistenceManager:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent / "saved"

        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.datasets_dir = self.base_dir / "datasets"
        self.manifest_path = self.base_dir / "manifest.json"
        self.warnings = []

    def _filename(self, name):
        """Nome legado, mantido para ler manifestos V1."""
        if not isinstance(name, str):
            raise TypeError("O nome deve ser texto.")

        encoded = base64.urlsafe_b64encode(
            name.encode("utf-8")
        ).decode("ascii").rstrip("=")

        return encoded + ".json"

    def _versioned_filename(self, name, generation):
        encoded = self._filename(name)[:-5]
        return encoded + "." + generation + ".json"

    @staticmethod
    def _safe_filename(filename):
        return (
            isinstance(filename, str)
            and bool(filename)
            and filename not in (".", "..")
            and Path(filename).name == filename
            and "/" not in filename
            and "\\" not in filename
            and filename.endswith(".json")
        )

    def _resolve_file(self, directory, filename):
        if not self._safe_filename(filename):
            raise ValueError("Nome de ficheiro persistido invalido.")
        return directory / filename

    @staticmethod
    def _valid_names(value, label):
        if not isinstance(value, list):
            raise ValueError("Lista de " + label + " invalida.")

        result = []
        seen = set()
        for name in value:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Nome invalido na lista de " + label + ".")
            if name in seen:
                raise ValueError("Nome duplicado na lista de " + label + ".")
            seen.add(name)
            result.append(name)
        return result

    def save_all(self, alpha, datasets):
        """Guarda modelos suportados e datasets; publica o manifesto no fim."""
        if not isinstance(datasets, dict):
            raise TypeError("datasets deve ser um dicionario.")

        self.warnings = []
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        generation = uuid.uuid4().hex
        model_names = []
        dataset_names = []
        model_files = {}
        dataset_files = {}
        model_metadata = {}
        created_files = []

        try:
            for name in alpha.list_models():
                model = alpha.get_model(name)

                # Objetos customizados nao sao serializados nem executados.
                if not isinstance(model, (Network, LanguageModel)):
                    self.warnings.append(
                        f"Modelo {name} ignorado: arquitetura sem checkpoint suportado."
                    )
                    continue

                model_filename = self._versioned_filename(name, generation)
                model_path = self.models_dir / model_filename

                metadata = None
                manager = getattr(alpha, "model_manager", None)
                if manager is not None and hasattr(manager, "export_metadata"):
                    metadata = manager.export_metadata(name)

                save_any(model, model_path, metadata=metadata)
                created_files.append(model_path)
                model_names.append(name)
                model_files[name] = model_filename

                if metadata is not None:
                    model_metadata[name] = metadata

                if name in datasets:
                    dataset_filename = self._versioned_filename(name, generation)
                    dataset_path = self.datasets_dir / dataset_filename
                    dataset_data = {
                        "format": "alpha_dataset_v1",
                        "samples": [
                            [list(inputs), list(targets)]
                            for inputs, targets in datasets[name]
                        ],
                    }
                    write_json_atomic(dataset_data, dataset_path)
                    created_files.append(dataset_path)
                    dataset_names.append(name)
                    dataset_files[name] = dataset_filename

            manifest = {
                "format": MANIFEST_V2,
                "format_version": MANIFEST_VERSION,
                "generation": generation,
                "models": model_names,
                "datasets": dataset_names,
                "model_files": model_files,
                "dataset_files": dataset_files,
                "model_metadata": model_metadata,
            }

            # O manifesto e o ponto de commit: só muda depois de todos os
            # checkpoints e datasets da nova geração estarem escritos.
            write_json_atomic(manifest, self.manifest_path)

        except BaseException:
            # Remove apenas ficheiros desta tentativa; nunca toca na geração
            # anteriormente publicada pelo manifesto.
            for path in created_files:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def _read_manifest(self):
        with open(self.manifest_path, "r", encoding="utf-8") as file:
            manifest = json.load(file)

        if not isinstance(manifest, dict):
            raise ValueError("Manifesto invalido.")

        manifest_format = manifest.get("format")
        if manifest_format == MANIFEST_V1:
            model_files = {}
            dataset_files = {}
            metadata = {}
        elif manifest_format == MANIFEST_V2:
            if manifest.get("format_version") != MANIFEST_VERSION:
                raise ValueError("Versao do manifesto nao suportada.")
            model_files = manifest.get("model_files")
            dataset_files = manifest.get("dataset_files")
            metadata = manifest.get("model_metadata", {})
            if not isinstance(model_files, dict):
                raise ValueError("Mapa de ficheiros de modelos invalido.")
            if not isinstance(dataset_files, dict):
                raise ValueError("Mapa de ficheiros de datasets invalido.")
            if not isinstance(metadata, dict):
                raise ValueError("Metadados de modelos invalidos.")
        else:
            raise ValueError("Formato de manifesto desconhecido.")

        model_names = self._valid_names(
            manifest.get("models", []), "modelos"
        )
        dataset_names = self._valid_names(
            manifest.get("datasets", []), "datasets"
        )

        return (
            manifest_format,
            model_names,
            dataset_names,
            model_files,
            dataset_files,
            metadata,
        )

    def load_all(self, alpha):
        """Recupera modelos Dense/Transformer e datasets; V1 continua suportado."""
        self.warnings = []
        datasets = {}

        if not self.manifest_path.exists():
            return datasets

        try:
            (
                manifest_format,
                model_names,
                dataset_names,
                model_files,
                dataset_files,
                metadata_by_name,
            ) = self._read_manifest()
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.warnings.append(
                "Nao foi possivel ler o manifesto: " + str(error)
            )
            return datasets

        for name in model_names:
            try:
                if name in alpha.list_models():
                    raise ValueError("Ja existe um modelo com esse nome.")

                if manifest_format == MANIFEST_V1:
                    filename = self._filename(name)
                else:
                    filename = model_files.get(name)
                    if filename is None:
                        raise ValueError("Falta o ficheiro do modelo no manifesto.")

                path = self._resolve_file(self.models_dir, filename)
                model = load_any(path)
                alpha.register_model(name, model)

                saved_metadata = metadata_by_name.get(name)
                if saved_metadata is not None:
                    manager = getattr(alpha, "model_manager", None)
                    if manager is None or not hasattr(manager, "restore_metadata"):
                        self.warnings.append(
                            f"Metadados de {name} ignorados: gestor incompativel."
                        )
                    else:
                        try:
                            manager.restore_metadata(name, saved_metadata)
                        except (ValueError, TypeError, KeyError) as error:
                            self.warnings.append(
                                f"Metadados de {name} nao restaurados: {error}"
                            )

            except (OSError, ValueError, TypeError, KeyError) as error:
                self.warnings.append(
                    f"Modelo {name} nao carregado: {error}"
                )

        for name in dataset_names:
            if name not in alpha.list_models():
                continue

            try:
                if manifest_format == MANIFEST_V1:
                    filename = self._filename(name)
                else:
                    filename = dataset_files.get(name)
                    if filename is None:
                        raise ValueError("Falta o ficheiro do dataset no manifesto.")

                path = self._resolve_file(self.datasets_dir, filename)
                with open(path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                if (
                    not isinstance(data, dict)
                    or data.get("format") != "alpha_dataset_v1"
                    or not isinstance(data.get("samples"), list)
                ):
                    raise ValueError("Formato de dataset invalido.")

                datasets[name] = Dataset(data["samples"])

            except (OSError, ValueError, TypeError, KeyError) as error:
                self.warnings.append(
                    f"Dataset {name} nao carregado: {error}"
                )

        return datasets
