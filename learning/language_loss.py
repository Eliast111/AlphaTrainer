"""Funcoes de perda para modelos de linguagem."""

import math


def cross_entropy(probabilities, targets):
    if not isinstance(probabilities, (list, tuple)):
        raise TypeError("Probabilidades devem ser uma lista.")
    if not isinstance(targets, (list, tuple)):
        raise TypeError("Alvos devem ser uma lista.")
    if not probabilities or len(probabilities) != len(targets):
        raise ValueError("Probabilidades e alvos devem ter o mesmo tamanho e nao estar vazios.")

    total_loss = 0.0

    for row, target in zip(probabilities, targets):
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError("Cada distribuicao deve ser uma lista nao vazia.")
        if isinstance(target, bool) or not isinstance(target, int):
            raise TypeError("Cada alvo deve ser um inteiro.")
        if target < 0 or target >= len(row):
            raise ValueError("Alvo fora dos limites do vocabulario.")

        values = [float(value) for value in row]
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError("As probabilidades devem ser finitas e estar entre zero e um.")

        total_loss -= math.log(max(values[target], 1e-12))

    return total_loss / len(targets)


def cross_entropy_derivative(probabilities, targets):
    if not isinstance(probabilities, (list, tuple)):
        raise TypeError("Probabilidades devem ser uma lista.")
    if not isinstance(targets, (list, tuple)):
        raise TypeError("Alvos devem ser uma lista.")
    if not probabilities or len(probabilities) != len(targets):
        raise ValueError("Probabilidades e alvos devem ter o mesmo tamanho e nao estar vazios.")

    gradients = []

    for row, target in zip(probabilities, targets):
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError("Cada distribuicao deve ser uma lista nao vazia.")
        if isinstance(target, bool) or not isinstance(target, int):
            raise TypeError("Cada alvo deve ser um inteiro.")
        if target < 0 or target >= len(row):
            raise ValueError("Alvo fora dos limites do vocabulario.")

        values = [float(value) for value in row]
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError("As probabilidades devem ser finitas e estar entre zero e um.")

        gradient = [0.0] * len(row)
        gradient[target] = -1.0 / (max(values[target], 1e-12) * len(targets))
        gradients.append(gradient)

    return gradients
