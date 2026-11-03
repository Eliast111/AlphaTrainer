"""
Alpha Evaluation Engine V0.1
Metricas para avaliar previsoes de modelos.
"""

import math


def _validate(predictions, targets):
    if not isinstance(predictions, (list, tuple)):
        raise TypeError("Predictions devem ser uma lista ou tupla.")

    if not isinstance(targets, (list, tuple)):
        raise TypeError("Targets devem ser uma lista ou tupla.")

    if not predictions or not targets:
        raise ValueError("As listas nao podem estar vazias.")

    if len(predictions) != len(targets):
        raise ValueError("Predictions e targets devem ter o mesmo tamanho.")

    try:
        predictions = [float(value) for value in predictions]
        targets = [float(value) for value in targets]
    except (TypeError, ValueError):
        raise ValueError("Todos os valores devem ser numericos.")

    if not all(math.isfinite(v) for v in predictions + targets):
        raise ValueError("Os valores devem ser numeros finitos.")

    return predictions, targets


def mse(predictions, targets):
    predictions, targets = _validate(predictions, targets)

    errors = [
        (prediction - target) ** 2
        for prediction, target in zip(predictions, targets)
    ]

    return sum(errors) / len(errors)


def mae(predictions, targets):
    predictions, targets = _validate(predictions, targets)

    errors = [
        abs(prediction - target)
        for prediction, target in zip(predictions, targets)
    ]

    return sum(errors) / len(errors)


def rmse(predictions, targets):
    return math.sqrt(mse(predictions, targets))


def evaluate(predictions, targets):
    predictions, targets = _validate(predictions, targets)

    return {
        "mse": mse(predictions, targets),
        "mae": mae(predictions, targets),
        "rmse": rmse(predictions, targets),
    }
