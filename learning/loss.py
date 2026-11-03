"""
Alpha Learning Engine V0.1
Funcoes de calculo de erro.
"""


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
