"""
Alpha Learning Engine V0.1
Funcoes de calculo de erro.
"""

import math

from alpha_math.backend import np, scipy_special


def mse(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("Predicoes e alvos devem ter o mesmo tamanho.")

    if not predictions:
        raise ValueError("As listas nao podem estar vazias.")

    errors = [
        (float(prediction) - float(target)) ** 2
        for prediction, target in zip(predictions, targets)
    ]

    return sum(errors) / len(errors)


def mse_derivative(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("Predicoes e alvos devem ter o mesmo tamanho.")

    if not predictions:
        raise ValueError("As listas nao podem estar vazias.")

    return [
        2.0 * (float(prediction) - float(target)) / len(predictions)
        for prediction, target in zip(predictions, targets)
    ]


def softmax(values):
    values = [float(v) for v in values]

    if not values:
        raise ValueError("A lista nao pode estar vazia.")

    if scipy_special is not None:
        return [float(v) for v in scipy_special.softmax(values)]

    if np is not None:
        shifted = np.asarray(values, dtype=float) - max(values)
        exponents = np.exp(shifted)
        return (exponents / exponents.sum()).astype(float).tolist()

    maximum = max(values)
    exponents = [math.exp(v - maximum) for v in values]
    total = sum(exponents)
    return [e / total for e in exponents]


def cross_entropy(probabilities, targets):
    """Entropia cruzada entre probabilidades previstas e alvos (one-hot)."""
    if len(probabilities) != len(targets):
        raise ValueError("Predicoes e alvos devem ter o mesmo tamanho.")

    if not probabilities:
        raise ValueError("As listas nao podem estar vazias.")

    return -sum(
        float(t) * math.log(max(float(p), 1e-12))
        for p, t in zip(probabilities, targets)
    )
