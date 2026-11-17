"""
Alpha Universal Evaluation Engine V1.0
Avaliacao unificada de diferentes arquiteturas de IA.
"""

from evaluation.evaluator import evaluate_model


class UniversalEvaluationError(RuntimeError):
    """Erro durante a avaliacao universal."""


class UniversalEvaluationEngine:
    def __init__(self, manager):
        self.manager = manager

    def evaluate(self, name, data):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Nome do modelo invalido.")

        model = self.manager.registry.get(name)
        adapter = self.manager.adapter_for(name)

        if not adapter.capabilities.evaluate:
            raise UniversalEvaluationError(
                "Este modelo nao suporta avaliacao."
            )

        architecture = adapter.architecture

        if architecture == "dense":
            result = evaluate_model(model, data)
            metrics = result["metrics"]
            samples = result["samples"]

        elif architecture == "transformer_decoder":
            evaluator = getattr(model, "evaluate", None)

            if not callable(evaluator):
                raise UniversalEvaluationError(
                    "O Transformer nao possui avaliador compativel."
                )

            result = evaluator(data)

            if not isinstance(result, dict):
                raise UniversalEvaluationError(
                    "O avaliador deve devolver um dicionario de metricas."
                )

            metrics = result
            samples = result.get("tokens")

        else:
            evaluator = getattr(model, "evaluate", None)

            if not callable(evaluator):
                raise UniversalEvaluationError(
                    "Este modelo nao possui avaliador compativel."
                )

            result = evaluator(data)

            if not isinstance(result, dict):
                raise UniversalEvaluationError(
                    "O avaliador deve devolver um dicionario de metricas."
                )

            metrics = result.get("metrics", result)
            samples = result.get("samples")

        evaluation = {
            "model": name,
            "architecture": architecture,
            "samples": samples,
            "metrics": metrics,
            "details": result,
        }

        self.manager.update_training(
            name,
            metrics={
                "evaluation": {
                    "architecture": architecture,
                    "metrics": metrics,
                }
            },
        )

        return evaluation
