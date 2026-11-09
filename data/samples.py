"""
Alpha Data Center - datasets de exemplo
"""

import math
import random

from data.dataset import Dataset


def _gate(function):
    return Dataset([
        ([float(a), float(b)], [float(function(a, b))])
        for a in (0, 1)
        for b in (0, 1)
    ])


def sine(points=24):
    """Regressao: aprender sin(x) em [0, pi]."""
    return Dataset([
        ([x], [math.sin(x)])
        for x in (math.pi * i / (points - 1) for i in range(points))
    ])


def linear(points=10):
    """Regressao linear simples: y = 2x + 1 (x em [0, 1])."""
    return Dataset([
        ([x], [2.0 * x + 1.0])
        for x in (i / (points - 1) for i in range(points))
    ])


BLOBS_CLASSES = ["A", "B", "C"]


def blobs(points_per_class=20, seed=0):
    """Classificacao: 3 grupos de pontos 2D (alvos one-hot)."""
    generator = random.Random(seed)
    centers = [(-1.0, -1.0), (1.0, -1.0), (0.0, 1.2)]
    samples = []

    for label, (cx, cy) in enumerate(centers):
        for _ in range(points_per_class):
            point = [generator.gauss(cx, 0.35), generator.gauss(cy, 0.35)]
            target = [1.0 if i == label else 0.0 for i in range(3)]
            samples.append((point, target))

    generator.shuffle(samples)
    return Dataset(samples)


# Datasets de classificacao e os nomes das suas classes.
CLASSIFICATION = {"blobs": BLOBS_CLASSES}

SAMPLES = {
    "xor": (lambda: _gate(lambda a, b: a ^ b), "Porta XOR (nao linear)", 2),
    "and": (lambda: _gate(lambda a, b: a & b), "Porta AND", 2),
    "or": (lambda: _gate(lambda a, b: a | b), "Porta OR", 2),
    "seno": (sine, "Regressao: sin(x) em [0, pi]", 1),
    "linear": (linear, "Regressao: y = 2x + 1", 1),
    "blobs": (blobs, "Classificacao: 3 grupos 2D (A, B, C)", 2),
}


def available_samples():
    return {name: info[1] for name, info in SAMPLES.items()}


def load_sample(name):
    key = str(name).strip().lower()

    if key not in SAMPLES:
        raise ValueError(
            "Dataset de exemplo desconhecido. Disponiveis: "
            + ", ".join(SAMPLES)
        )

    return SAMPLES[key][0]()
