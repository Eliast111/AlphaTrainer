"""
Alpha Persistence Engine V1.0
Armazenamento permanente de modelos e datasets.
"""

import base64
import json
import os
from pathlib import Path

from checkpoints.checkpoint import save_model, load_model
from data.dataset import Dataset


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
        encoded = base64.urlsafe_b64encode(
            name.encode("utf-8")
        ).decode("ascii").rstrip("=")

        return encoded + ".json"

    def save_all(self, alpha, datasets):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        model_names = alpha.list_models()
        dataset_names = []

        for name in model_names:
            model_path = self.models_dir / self._filename(name)
            save_model(alpha.get_model(name), model_path)

            if name in datasets:
                dataset_path = (
                    self.datasets_dir / self._filename(name)
                )

                data = {
                    "format": "alpha_dataset_v1",
                    "samples": [
                        [list(inputs), list(targets)]
                        for inputs, targets in datasets[name]
                    ]
                }

                temporary_path = dataset_path.with_suffix(".tmp")

                with open(
                    temporary_path, "w", encoding="utf-8"
                ) as file:
                    json.dump(data, file, indent=4)

                os.replace(temporary_path, dataset_path)
                dataset_names.append(name)

        manifest = {
            "format": "alpha_persistence_v1",
            "models": model_names,
            "datasets": dataset_names
        }

        temporary_manifest = self.manifest_path.with_suffix(".tmp")

        with open(
            temporary_manifest, "w", encoding="utf-8"
        ) as file:
            json.dump(manifest, file, indent=4)

        os.replace(temporary_manifest, self.manifest_path)

    def load_all(self, alpha):
        self.warnings = []
        datasets = {}

        if not self.manifest_path.exists():
            return datasets

        try:
            with open(
                self.manifest_path, "r", encoding="utf-8"
            ) as file:
                manifest = json.load(file)

            if (
                not isinstance(manifest, dict)
                or manifest.get("format") != "alpha_persistence_v1"
            ):
                raise ValueError("Manifesto invalido.")

            model_names = manifest.get("models", [])
            dataset_names = manifest.get("datasets", [])

            if not isinstance(model_names, list):
                raise ValueError("Lista de modelos invalida.")

            if not isinstance(dataset_names, list):
                raise ValueError("Lista de datasets invalida.")

        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.warnings.append(
                "Nao foi possivel ler o manifesto: " + str(error)
            )
            return datasets

        for name in model_names:
            if not isinstance(name, str) or not name.strip():
                self.warnings.append("Nome de modelo invalido.")
                continue

            try:
                path = self.models_dir / self._filename(name)
                model = load_model(path)
                alpha.register_model(name, model)

            except (OSError, ValueError, TypeError, KeyError) as error:
                self.warnings.append(
                    f"Modelo {name} nao carregado: {error}"
                )

        for name in dataset_names:
            if not isinstance(name, str) or name not in alpha.list_models():
                continue

            try:
                path = self.datasets_dir / self._filename(name)

                with open(path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                if (
                    not isinstance(data, dict)
                    or data.get("format") != "alpha_dataset_v1"
                ):
                    raise ValueError("Formato de dataset invalido.")

                datasets[name] = Dataset(data["samples"])

            except (OSError, ValueError, TypeError, KeyError) as error:
                self.warnings.append(
                    f"Dataset {name} nao carregado: {error}"
                )

        return datasets
