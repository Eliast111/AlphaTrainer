"""
Alpha Checkpoint Engine - Language Model
Guardar e carregar modelos de linguagem (vocabulario + pesos) em JSON.
"""

import json
import math

from checkpoints.checkpoint import write_json_atomic
from language.vocabulary import Vocabulary
from transformer.language_model import LanguageModel

FORMAT = "alpha_lm_checkpoint_v1"


def _check_finite(array):
    rows = array if array and isinstance(array[0], list) else [array]

    for row in rows:
        for value in row:
            if not math.isfinite(value):
                raise ValueError("O modelo contem valores nao finitos.")


def save_language_model(model, path):
    if not isinstance(model, LanguageModel):
        raise TypeError("O objeto deve ser um LanguageModel.")

    weights = {}

    for name, array in model.parameters():
        _check_finite(array)
        weights[name] = array

    data = {
        "format": FORMAT,
        "config": {
            "embedding_dim": model.embedding_dim,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "max_length": model.max_length,
        },
        "vocabulary": list(model.vocabulary.id_to_token),
        "weights": weights,
    }

    write_json_atomic(data, path)


def _shape_matches(array, rows, cols):
    return (
        isinstance(array, list)
        and len(array) == rows
        and all(isinstance(r, list) and len(r) == cols for r in array)
    )


def load_language_model(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError("Formato de checkpoint de linguagem desconhecido.")

    config = data.get("config")
    tokens = data.get("vocabulary")
    weights = data.get("weights")

    if not isinstance(config, dict) or not isinstance(weights, dict):
        raise ValueError("Checkpoint de linguagem invalido.")

    if (
        not isinstance(tokens, list)
        or tokens[:len(Vocabulary.SPECIAL_TOKENS)]
        != list(Vocabulary.SPECIAL_TOKENS)
    ):
        raise ValueError("Vocabulario invalido no checkpoint.")

    vocabulary = Vocabulary()

    for token in tokens:
        vocabulary.add_token(token)

    if vocabulary.size != len(tokens):
        raise ValueError("Vocabulario com tokens repetidos.")

    model = LanguageModel(
        vocabulary,
        embedding_dim=config.get("embedding_dim"),
        hidden_dim=config.get("hidden_dim"),
        num_layers=config.get("num_layers"),
        max_length=config.get("max_length"),
        seed=0
    )

    for name, current in model.parameters():
        if name not in weights:
            raise ValueError(f"Falta o parametro {name!r}.")

        saved = weights[name]
        is_matrix = bool(current) and isinstance(current[0], list)

        if is_matrix:
            if not _shape_matches(saved, len(current), len(current[0])):
                raise ValueError(f"Dimensoes invalidas em {name!r}.")
        elif not isinstance(saved, list) or len(saved) != len(current):
            raise ValueError(f"Dimensoes invalidas em {name!r}.")

        try:
            if is_matrix:
                values = [[float(v) for v in row] for row in saved]
            else:
                values = [float(v) for v in saved]
        except (TypeError, ValueError):
            raise ValueError(f"Valores invalidos em {name!r}.")

        _check_finite(values)

        if is_matrix:
            for row, new_row in zip(current, values):
                row[:] = new_row
        else:
            current[:] = values

    return model
