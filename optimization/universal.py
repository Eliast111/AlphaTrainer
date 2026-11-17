"""
Universal Model Optimization Engine V1.0
Experiencias independentes para otimizacao de modelos.
"""

import math
from copy import deepcopy

from learning.optimizer import create_optimizer
from training.universal import UniversalTrainingEngine
from evaluation.universal import UniversalEvaluationEngine


class OptimizationError(RuntimeError):
    """Erro durante a otimizacao de modelos."""


class UniversalOptimizationEngine:

    ARCHITECTURE_METRICS = {
        "dense": {"mse", "mae", "rmse"},
        "transformer_decoder": {"loss", "perplexity"},
    }

    VALID_METRICS = {
        metric
        for metrics in ARCHITECTURE_METRICS.values()
        for metric in metrics
    }

    VALID_DIRECTIONS = {"min", "max"}

    def __init__(self, manager):
        if manager is None:
            raise OptimizationError("Model manager invalido.")

        self.manager = manager
        self.training_engine = UniversalTrainingEngine(manager)
        self.evaluation_engine = UniversalEvaluationEngine(manager)

    def _temporary_name(self, name, index):
        base = "__alphaopt_" + name + "_" + str(index)
        candidate = base
        suffix = 1

        while candidate in self.manager.registry.models:
            candidate = base + "_" + str(suffix)
            suffix += 1

        return candidate

    def optimize(
        self,
        name,
        data,
        configurations,
        metric="mse",
        direction="min",
        epochs=100,
        **training_options
    ):
        if not isinstance(name, str) or not name.strip():
            raise OptimizationError("Nome do modelo invalido.")

        if not isinstance(configurations, (list, tuple)) or not configurations:
            raise OptimizationError("Lista de configuracoes vazia ou invalida.")

        if metric not in self.VALID_METRICS:
            raise OptimizationError("Metrica invalida: " + str(metric))

        if direction not in self.VALID_DIRECTIONS:
            raise OptimizationError("Direcao de classificacao invalida.")

        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
            raise OptimizationError("Numero de epochs invalido.")

        try:
            adapter = self.manager.adapter_for(name)
        except Exception as exc:
            raise OptimizationError("Modelo original nao encontrado.") from exc

        architecture = adapter.architecture

        if not adapter.capabilities.train or not adapter.capabilities.evaluate:
            raise OptimizationError("Modelo nao suporta treino e avaliacao.")

        if architecture not in self.ARCHITECTURE_METRICS:
            raise OptimizationError("Arquitetura ainda nao suportada.")

        available_metrics = self.ARCHITECTURE_METRICS[architecture]

        if metric not in available_metrics:
            raise OptimizationError(
                "Metrica '" + str(metric) +
                "' nao suportada por '" + architecture +
                "'. Metricas disponiveis: " +
                ", ".join(sorted(available_metrics))
            )

        validated = []

        for config in configurations:
            if not isinstance(config, dict):
                raise OptimizationError("Configuracao invalida.")

            config = deepcopy(config)
            learning_rate = config.get("learning_rate", 0.01)

            if (
                isinstance(learning_rate, bool)
                or not isinstance(learning_rate, (int, float))
                or not math.isfinite(learning_rate)
                or learning_rate <= 0
            ):
                raise OptimizationError("Learning rate invalido.")

            if architecture == "dense":
                optimizer_name = config.get("optimizer", "sgd")
                optimizer_options = {
                    key: value
                    for key, value in config.items()
                    if key not in ("optimizer", "learning_rate")
                }

                try:
                    create_optimizer(
                        name=optimizer_name,
                        learning_rate=learning_rate,
                        **optimizer_options
                    )
                except Exception as exc:
                    raise OptimizationError(
                        "Configuracao de optimizer invalida."
                    ) from exc

            elif "optimizer" in config:
                raise OptimizationError(
                    "Transformers utilizam learning_rate diretamente."
                )

            validated.append(config)

        results = []

        for index, config in enumerate(validated):
            temporary_name = self._temporary_name(name, index)
            registered = False

            try:
                self.manager.duplicate(name, temporary_name)
                registered = True

                if architecture == "dense":
                    optimizer_options = {
                        key: value
                        for key, value in config.items()
                        if key not in ("optimizer", "learning_rate")
                    }

                    optimizer = create_optimizer(
                        name=config.get("optimizer", "sgd"),
                        learning_rate=config.get("learning_rate", 0.01),
                        **optimizer_options
                    )

                    train_result = self.training_engine.train(
                        temporary_name,
                        data,
                        optimizer=optimizer,
                        epochs=epochs,
                        **training_options
                    )

                else:
                    train_result = self.training_engine.train(
                        temporary_name,
                        data,
                        epochs=epochs,
                        learning_rate=config.get("learning_rate", 0.01),
                        **training_options
                    )

                evaluation = self.evaluation_engine.evaluate(
                    temporary_name,
                    data
                )

                metrics = evaluation.get("metrics", {})
                value = metrics.get(metric)

                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise OptimizationError(
                        "Metrica indisponivel ou invalida: " + metric
                    )

                results.append({
                    "experiment": index + 1,
                    "configuration": deepcopy(config),
                    "metric": metric,
                    "value": value,
                    "training": train_result,
                    "evaluation": evaluation,
                    "status": "completed",
                })

            except Exception as exc:
                if isinstance(exc, OptimizationError):
                    raise
                raise OptimizationError(
                    "Falha na experiencia " + str(index + 1)
                ) from exc

            finally:
                if temporary_name in self.manager.registry.models:
                    import sys
                    active_error = sys.exc_info()[1]

                    try:
                        self.manager.remove(temporary_name)
                    except Exception as cleanup_error:
                        if active_error is not None:
                            if hasattr(active_error, "add_note"):
                                active_error.add_note(
                                    "Falha adicional na limpeza do modelo "
                                    + temporary_name + ": "
                                    + str(cleanup_error)
                                )
                        else:
                            raise OptimizationError(
                                "Falha ao remover o modelo temporario: "
                                + temporary_name
                            ) from cleanup_error

        ranking = sorted(
            results,
            key=lambda item: item["value"],
            reverse=(direction == "max")
        )

        return {
            "model": name,
            "architecture": architecture,
            "metric": metric,
            "direction": direction,
            "total_experiments": len(results),
            "completed_experiments": sum(
                1 for item in results
                if item["status"] == "completed"
            ),
            "experiments": results,
            "ranking": ranking,
            "best_configuration": deepcopy(
                ranking[0]["configuration"]
            ),
            "best_value": ranking[0]["value"],
        }
