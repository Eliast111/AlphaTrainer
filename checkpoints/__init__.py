from checkpoints.checkpoint import save_model, load_model, read_metadata
from checkpoints.lm_checkpoint import (
    save_language_model, load_language_model
)

__all__ = [
    "save_model", "load_model", "read_metadata",
    "save_language_model", "load_language_model",
    "load_any", "save_any",
]


def _format_of(path):
    import json

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data.get("format") if isinstance(data, dict) else None


def load_any(path):
    """Carrega uma rede neuronal ou um modelo de linguagem."""
    kind = _format_of(path)

    if kind == "alpha_checkpoint_v1":
        return load_model(path)

    if kind == "alpha_lm_checkpoint_v1":
        return load_language_model(path)

    raise ValueError("Formato de checkpoint desconhecido.")


def save_any(model, path, metadata=None):
    from neural.network import Network

    if isinstance(model, Network):
        return save_model(model, path, metadata=metadata)

    return save_language_model(model, path)
