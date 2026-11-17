"""Persistencia atomica de sessoes de treino."""

import json
import os

from checkpoints.checkpoint import write_json_atomic


FORMAT = "alpha_training_session_v1"


def save_training_session(path, payload):
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("O caminho do checkpoint deve ser texto.")

    if not isinstance(payload, dict):
        raise TypeError("O checkpoint deve ser um dicionario.")

    data = dict(payload)
    data["format"] = FORMAT

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    write_json_atomic(data, path)


def load_training_session(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Checkpoint de treino invalido.")

    if data.get("format") != FORMAT:
        raise ValueError("Formato de checkpoint de treino desconhecido.")

    return data
