"""
Alpha Universal Model Benchmark Engine V1.0
Comparacao de modelos usando metricas compativeis.
"""

import math

from evaluation.universal import UniversalEvaluationEngine


class BenchmarkError(RuntimeError):
    """Erro durante a comparacao de modelos."""


class UniversalBenchmarkEngine:

    def __init__(self, manager):
        self.evaluator = UniversalEvaluationEngine(manager)

    def run(self, models, metric="mse", direction="min"):
        if not isinstance(models, dict) or len(models) < 2:
            raise ValueError(
                "Forneca pelo menos dois modelos com os respetivos dados."
            )

        if not isinstance(metric, str) or not metric.strip():
            raise ValueError("Indique uma metrica valida.")

        if direction not in ("min", "max"):
            raise ValueError("A direcao deve ser 'min' ou 'max'.")

        results = {}

        for name, data in models.items():
            evaluation = self.evaluator.evaluate(name, data)
            metrics = evaluation["metrics"]

            if metric not in metrics:
                raise BenchmarkError(
                    "Metrica '" + metric + "' indisponivel no modelo "
                    + name
                )

            value = metrics[metric]

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise BenchmarkError(
                    "Metrica invalida no modelo " + name
                )

            results[name] = {
                "value": value,
                "architecture": evaluation["architecture"],
                "evaluation": evaluation,
            }

        ranking = sorted(
            results,
            key=lambda name: results[name]["value"],
            reverse=(direction == "max"),
        )

        return {
            "metric": metric,
            "direction": direction,
            "results": results,
            "ranking": ranking,
            "best_model": ranking[0],
            "best_value": results[ranking[0]]["value"],
        }
