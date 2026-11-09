"""
Alpha Data Center - pre-processamento
Normalizacao de entradas (e opcionalmente alvos) para treinar melhor.
"""

import math

from data.dataset import Dataset

METHODS = ("standard", "minmax")


def _stats(columns, method):
    result = []

    for column in columns:
        if method == "standard":
            mean = sum(column) / len(column)
            variance = sum((v - mean) ** 2 for v in column) / len(column)
            spread = math.sqrt(variance)
            result.append((mean, spread if spread > 1e-12 else 1.0))
        else:
            low, high = min(column), max(column)
            result.append((low, (high - low) if high - low > 1e-12 else 1.0))

    return result


class Scaler:
    """Normaliza entradas (e alvos, se scale_targets=True).

    standard: (x - media) / desvio padrao
    minmax:   (x - minimo) / (maximo - minimo)  -> [0, 1]
    """

    def __init__(self, method="standard", scale_targets=False):
        if method not in METHODS:
            raise ValueError(
                "Metodo desconhecido. Disponiveis: " + ", ".join(METHODS)
            )

        self.method = method
        self.scale_targets = bool(scale_targets)
        self.input_stats = None
        self.target_stats = None

    @property
    def fitted(self):
        return self.input_stats is not None

    def fit(self, dataset):
        if not isinstance(dataset, Dataset):
            raise TypeError("fit exige um Dataset.")

        inputs = list(zip(*[sample[0] for sample in dataset]))
        self.input_stats = _stats(inputs, self.method)

        if self.scale_targets:
            targets = list(zip(*[sample[1] for sample in dataset]))
            self.target_stats = _stats(targets, self.method)

        return self

    def _require_fit(self):
        if not self.fitted:
            raise ValueError("O Scaler ainda nao foi ajustado (fit).")

    def transform_inputs(self, values):
        self._require_fit()
        values = list(values)

        if len(values) != len(self.input_stats):
            raise ValueError("Numero de entradas incompativel com o Scaler.")

        return [
            (float(v) - offset) / spread
            for v, (offset, spread) in zip(values, self.input_stats)
        ]

    def transform_targets(self, values):
        if self.target_stats is None:
            return [float(v) for v in values]

        return [
            (float(v) - offset) / spread
            for v, (offset, spread) in zip(values, self.target_stats)
        ]

    def inverse_targets(self, values):
        if self.target_stats is None:
            return list(values)

        return [
            v * spread + offset
            for v, (offset, spread) in zip(values, self.target_stats)
        ]

    def transform(self, dataset):
        self._require_fit()

        return Dataset([
            (self.transform_inputs(inputs), self.transform_targets(targets))
            for inputs, targets in dataset
        ])

    def to_dict(self):
        self._require_fit()

        return {
            "method": self.method,
            "scale_targets": self.scale_targets,
            "input_stats": [list(s) for s in self.input_stats],
            "target_stats": (
                None if self.target_stats is None
                else [list(s) for s in self.target_stats]
            ),
        }

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Dados do Scaler invalidos.")

        scaler = cls(data.get("method", "standard"),
                     data.get("scale_targets", False))

        def read(stats):
            result = []

            for pair in stats:
                offset, spread = float(pair[0]), float(pair[1])

                if not (math.isfinite(offset) and math.isfinite(spread)) \
                        or spread == 0:
                    raise ValueError("Estatisticas do Scaler invalidas.")

                result.append((offset, spread))

            return result

        try:
            scaler.input_stats = read(data["input_stats"])

            if data.get("target_stats") is not None:
                scaler.target_stats = read(data["target_stats"])
        except (KeyError, TypeError, IndexError):
            raise ValueError("Dados do Scaler invalidos.")

        return scaler


class ScaledModel:
    """Embrulha um modelo para receber dados originais (nao normalizados)."""

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def forward(self, inputs):
        output = self.model.forward(self.scaler.transform_inputs(inputs))
        return self.scaler.inverse_targets(output)

    predict = forward
