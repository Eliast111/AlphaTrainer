"""
Alpha Data Center V0.2
Validacao, organizacao e importacao de datasets.
"""

import csv
import math
import random


class Dataset:
    def __init__(self, samples):
        if not samples:
            raise ValueError("O dataset nao pode estar vazio.")

        self.samples = []
        input_size = None
        target_size = None

        for sample in samples:
            if not isinstance(sample, (list, tuple)) or len(sample) != 2:
                raise ValueError("Cada amostra deve conter entradas e alvos.")

            inputs, targets = sample

            if not isinstance(inputs, (list, tuple)):
                raise TypeError("As entradas devem ser listas ou tuplas.")

            if not isinstance(targets, (list, tuple)):
                raise TypeError("Os alvos devem ser listas ou tuplas.")

            if not inputs or not targets:
                raise ValueError("Entradas e alvos nao podem estar vazios.")

            try:
                inputs = [float(value) for value in inputs]
                targets = [float(value) for value in targets]
            except (TypeError, ValueError):
                raise ValueError("Os dados devem ser numericos.")

            if not all(math.isfinite(v) for v in inputs + targets):
                raise ValueError("Os dados devem conter numeros finitos.")

            if input_size is None:
                input_size = len(inputs)
                target_size = len(targets)

            if len(inputs) != input_size:
                raise ValueError("Tamanho das entradas inconsistente.")

            if len(targets) != target_size:
                raise ValueError("Tamanho dos alvos inconsistente.")

            self.samples.append((inputs, targets))

        self.input_size = input_size
        self.target_size = target_size

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def split(self, validation_ratio=0.2, shuffle=True, seed=None):
        ratio = float(validation_ratio)

        if not math.isfinite(ratio) or not 0 < ratio < 1:
            raise ValueError("A proporcao deve estar entre 0 e 1.")

        if len(self.samples) < 2:
            raise ValueError("Sao necessarias pelo menos duas amostras.")

        samples = list(self.samples)

        if shuffle:
            random.Random(seed).shuffle(samples)

        validation_count = max(1, math.ceil(len(samples) * ratio))
        validation_count = min(validation_count, len(samples) - 1)

        validation = samples[:validation_count]
        training = samples[validation_count:]

        return Dataset(training), Dataset(validation)

    @classmethod
    def from_csv(
        cls,
        path,
        target_columns=1,
        delimiter=",",
        has_header=True
    ):
        if (
            not isinstance(target_columns, int)
            or isinstance(target_columns, bool)
            or target_columns < 1
        ):
            raise ValueError("target_columns deve ser um inteiro positivo.")

        if not isinstance(delimiter, str) or len(delimiter) != 1:
            raise ValueError("O delimitador deve ter um unico caractere.")

        samples = []

        with open(path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file, delimiter=delimiter)

            if has_header:
                next(reader, None)

            start_line = 2 if has_header else 1

            for line_number, row in enumerate(reader, start=start_line):
                if not row or all(not cell.strip() for cell in row):
                    continue

                try:
                    values = [float(cell.strip()) for cell in row]
                except ValueError:
                    raise ValueError(
                        f"Valor nao numerico na linha {line_number}."
                    )

                if not all(math.isfinite(v) for v in values):
                    raise ValueError(
                        f"Numero invalido na linha {line_number}."
                    )

                if len(values) <= target_columns:
                    raise ValueError(
                        f"Colunas insuficientes na linha {line_number}."
                    )

                inputs = values[:-target_columns]
                targets = values[-target_columns:]

                samples.append((inputs, targets))

        if not samples:
            raise ValueError("O ficheiro CSV nao contem dados validos.")

        return cls(samples)
