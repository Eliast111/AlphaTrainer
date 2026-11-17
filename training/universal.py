"""Motor universal de treino do AlphaTrainer."""

from training.trainer import Trainer


class UniversalTrainingError(RuntimeError):
    """Erro durante o treino universal."""


class UniversalTrainingEngine:
    def __init__(self, manager):
        self.manager = manager

    def train(self, name, data, optimizer=None, epochs=100, **options):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Nome do modelo invalido.")

        model = self.manager.registry.get(name)
        adapter = self.manager.adapter_for(name)
        capabilities = adapter.capabilities

        if not capabilities.train:
            raise UniversalTrainingError(
                "Este modelo nao suporta treino."
            )

        architecture = adapter.architecture

        if options.get("resume") or options.get("checkpoint_path") is not None:
            if not capabilities.resume:
                raise UniversalTrainingError(
                    "Este modelo nao suporta checkpoints de sessao nem retoma local."
                )

        self.manager.update_training(name, status="training")

        try:
            if architecture == "dense":
                if optimizer is None:
                    raise ValueError(
                        "O treino de redes densas requer um optimizer."
                    )

                trainer = Trainer(optimizer)
                result = trainer.fit(
                    model,
                    data,
                    epochs=epochs,
                    **options
                )

            elif architecture == "transformer_decoder":
                result = model.fit(
                    data,
                    epochs=epochs,
                    **options
                )

            else:
                fit = getattr(model, "fit", None)

                if not callable(fit):
                    raise UniversalTrainingError(
                        "O modelo nao possui um metodo fit compativel."
                    )

                result = fit(
                    data,
                    epochs=epochs,
                    **options
                )

            self.manager.update_training(
                name,
                status="completed",
                metrics=result,
            )

            return result

        except Exception:
            self.manager.update_training(
                name,
                status="failed",
            )
            raise
