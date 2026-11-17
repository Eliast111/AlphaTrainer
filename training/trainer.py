"""
Alpha Training Engine V0.3
Treinamento com mini-batches, validacao, early stopping,
gradient clipping, callbacks e checkpoints de sessao.
"""

import copy
import hashlib
import json
import math
import random

from learning.backprop import backpropagate
from learning.loss import mse, cross_entropy, softmax
from training.session_checkpoint import (
    save_training_session,
    load_training_session,
)


def _argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def _loss_function(loss):
    if loss == "mse":
        return mse

    if loss == "cross_entropy":
        return cross_entropy

    raise ValueError("loss deve ser 'mse' ou 'cross_entropy'.")


def _is_positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _average_gradients(batch_gradients):
    """Faz a media de varios conjuntos de gradientes."""
    count = float(len(batch_gradients))
    first = batch_gradients[0]
    averaged = []

    for layer_index, layer in enumerate(first):
        layer_result = []

        for neuron_index in range(len(layer)):
            weights = [0.0] * len(layer[neuron_index]["weights"])
            bias = 0.0

            for gradients in batch_gradients:
                item = gradients[layer_index][neuron_index]

                for i, value in enumerate(item["weights"]):
                    weights[i] += value

                bias += item["bias"]

            layer_result.append({
                "weights": [w / count for w in weights],
                "bias": bias / count,
            })

        averaged.append(layer_result)

    return averaged


def _clip_gradients(gradients, max_norm):
    squares = 0.0

    for layer in gradients:
        for item in layer:
            squares += sum(w * w for w in item["weights"])
            squares += item["bias"] ** 2

    norm = math.sqrt(squares)

    if norm <= max_norm or norm == 0.0:
        return gradients

    factor = max_norm / norm

    return [
        [
            {
                "weights": [w * factor for w in item["weights"]],
                "bias": item["bias"] * factor,
            }
            for item in layer
        ]
        for layer in gradients
    ]


def _snapshot(network):
    return [
        [(list(n.weights), n.bias) for n in layer.neurons]
        for layer in network.layers
    ]


def _restore(network, state):
    if not isinstance(state, list) or len(state) != len(network.layers):
        raise ValueError("Arquitetura do checkpoint incompativel.")

    for layer, layer_state in zip(network.layers, state):
        if not isinstance(layer_state, list):
            raise ValueError("Estado de camada invalido.")

        if len(layer_state) != len(layer.neurons):
            raise ValueError("Numero de neuronios incompativel.")

        for neuron, saved in zip(layer.neurons, layer_state):
            if not isinstance(saved, (list, tuple)) or len(saved) != 2:
                raise ValueError("Estado de neuronio invalido.")

            weights, bias = saved

            if not isinstance(weights, (list, tuple)):
                raise ValueError("Lista de pesos invalida.")

            if len(weights) != len(neuron.weights):
                raise ValueError("Numero de pesos incompativel.")

            values = list(weights) + [bias]

            if any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                for v in values
            ):
                raise ValueError("Checkpoint contem valores invalidos.")

    for layer, layer_state in zip(network.layers, state):
        for neuron, (weights, bias) in zip(layer.neurons, layer_state):
            neuron.weights = list(weights)
            neuron.bias = bias


def _rng_to_json(value):
    if isinstance(value, tuple):
        return [_rng_to_json(item) for item in value]

    return value


def _rng_from_json(value):
    if isinstance(value, list):
        return tuple(_rng_from_json(item) for item in value)

    return value


def _dataset_signature(samples):
    serialized = json.dumps(
        samples,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _setting_signature(value):
    return repr(value)


def _network_shape(network):
    return [
        [len(neuron.weights) for neuron in layer.neurons]
        for layer in network.layers
    ]


def _validate_history(history, name):
    if not isinstance(history, list):
        raise ValueError(f"Historico {name} invalido.")

    for value in history:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Historico {name} contem valores invalidos.")


class Trainer:
    def __init__(self, optimizer):
        if not callable(getattr(optimizer, "step", None)):
            raise TypeError("Optimizer invalido.")

        self.optimizer = optimizer

    @staticmethod
    def _prepare(network, dataset, label="dataset", loss="mse"):
        if not dataset:
            raise ValueError(f"O {label} nao pode estar vazio.")

        samples = []

        for sample in dataset:
            if not isinstance(sample, (list, tuple)) or len(sample) != 2:
                raise ValueError("Cada amostra deve conter entradas e alvos.")

            inputs, targets = sample

            if not inputs or not targets:
                raise ValueError("Entradas e alvos nao podem estar vazios.")

            inputs = list(inputs)
            targets = list(targets)

            prediction = network.forward(inputs)

            if len(prediction) != len(targets):
                raise ValueError("Tamanho dos alvos incompativel.")

            if loss == "cross_entropy":
                if any(t < 0 for t in targets):
                    raise ValueError("Alvos de classificacao devem ser >= 0.")

                cross_entropy(softmax(prediction), targets)
            else:
                mse(prediction, targets)

            samples.append((inputs, targets))

        return samples

    @staticmethod
    def _loss(network, samples, loss="mse"):
        if loss == "cross_entropy":
            return sum(
                cross_entropy(softmax(network.forward(i)), t)
                for i, t in samples
            ) / len(samples)

        return sum(
            mse(network.forward(inputs), targets)
            for inputs, targets in samples
        ) / len(samples)

    @staticmethod
    def _accuracy(network, samples):
        hits = sum(
            _argmax(network.forward(i)) == _argmax(t) for i, t in samples
        )
        return hits / len(samples)

    def fit(
        self,
        network,
        dataset,
        epochs=100,
        batch_size=1,
        shuffle=True,
        seed=0,
        validation_data=None,
        patience=None,
        min_delta=0.0,
        restore_best=True,
        clip_norm=None,
        target_loss=None,
        callback=None,
        loss="mse",
        checkpoint_path=None,
        checkpoint_every=1,
        resume=False,
    ):
        """Treina, guarda checkpoints e permite retomar sessoes.

        epochs: total de epocas pretendido, incluindo epocas anteriores.
        checkpoint_path: caminho opcional para guardar a sessao.
        checkpoint_every: intervalo de epocas entre checkpoints.
        resume: retoma a sessao guardada em checkpoint_path.
        """

        loss_fn = _loss_function(loss)

        if not _is_positive_int(epochs):
            raise ValueError("Epochs deve ser um inteiro positivo.")

        if not _is_positive_int(batch_size):
            raise ValueError("batch_size deve ser um inteiro positivo.")

        if patience is not None and not _is_positive_int(patience):
            raise ValueError("patience deve ser um inteiro positivo ou None.")

        if not _is_positive_int(checkpoint_every):
            raise ValueError("checkpoint_every deve ser um inteiro positivo.")

        if clip_norm is not None and not (
            isinstance(clip_norm, (int, float))
            and not isinstance(clip_norm, bool)
            and math.isfinite(clip_norm)
            and clip_norm > 0
        ):
            raise ValueError("clip_norm deve ser positivo.")

        if callback is not None and not callable(callback):
            raise TypeError("callback deve ser uma funcao.")

        if not isinstance(resume, bool):
            raise TypeError("resume deve ser True ou False.")

        if resume and checkpoint_path is None:
            raise ValueError("resume requer checkpoint_path.")

        samples = self._prepare(network, dataset, loss=loss)

        validation = None

        if validation_data is not None:
            validation = self._prepare(
                network, validation_data, "dataset de validacao", loss=loss
            )

        if checkpoint_path is not None:
            if not callable(getattr(self.optimizer, "state_dict", None)):
                raise TypeError("Optimizer nao suporta state_dict.")

            if not callable(getattr(self.optimizer, "load_state_dict", None)):
                raise TypeError("Optimizer nao suporta load_state_dict.")

        config = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "seed": _setting_signature(seed),
            "patience": patience,
            "min_delta": _setting_signature(min_delta),
            "restore_best": restore_best,
            "clip_norm": _setting_signature(clip_norm),
            "target_loss": _setting_signature(target_loss),
            "loss": loss,
        }

        dataset_hash = _dataset_signature(samples)
        validation_hash = (
            _dataset_signature(validation) if validation is not None else None
        )
        shape = _network_shape(network)

        rng = random.Random(seed)
        history = []
        val_history = []
        accuracy_history = []
        val_accuracy_history = []
        best_loss = math.inf
        best_epoch = 0
        best_state = None
        stale = 0
        stopped_early = False
        order = list(range(len(samples)))
        start_epoch = 0

        if resume:
            saved = load_training_session(checkpoint_path)

            if saved.get("config") != config:
                raise ValueError("Configuracoes do treino incompativeis.")

            if saved.get("dataset_signature") != dataset_hash:
                raise ValueError("O dataset nao corresponde ao checkpoint.")

            if saved.get("validation_signature") != validation_hash:
                raise ValueError("Os dados de validacao nao correspondem.")

            if saved.get("network_shape") != shape:
                raise ValueError("A arquitetura da rede nao corresponde.")

            start_epoch = saved.get("epoch")

            if (
                isinstance(start_epoch, bool)
                or not isinstance(start_epoch, int)
                or start_epoch < 1
                or start_epoch >= epochs
            ):
                raise ValueError(
                    "O total de epochs deve ser superior a epoca guardada."
                )

            history = saved.get("history")
            val_history = saved.get("val_history")
            accuracy_history = saved.get("accuracy_history")
            val_accuracy_history = saved.get("val_accuracy_history")

            _validate_history(history, "de treino")
            _validate_history(val_history, "de validacao")
            _validate_history(accuracy_history, "de acuracia")
            _validate_history(val_accuracy_history, "de acuracia de validacao")

            if len(history) != start_epoch:
                raise ValueError("Historico de treino inconsistente.")

            if validation is not None and len(val_history) != start_epoch:
                raise ValueError("Historico de validacao inconsistente.")

            if loss == "cross_entropy" and len(accuracy_history) != start_epoch:
                raise ValueError("Historico de acuracia inconsistente.")

            if loss == "cross_entropy" and validation is not None:
                if len(val_accuracy_history) != start_epoch:
                    raise ValueError("Historico de validacao inconsistente.")

            order = saved.get("order")

            if (
                not isinstance(order, list)
                or sorted(order) != list(range(len(samples)))
            ):
                raise ValueError("Ordem das amostras invalida.")

            if not isinstance(order, list) or any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in order
            ) or sorted(order) != list(range(len(samples))):
                raise ValueError("Ordem das amostras invalida.")

            if validation is None and val_history:
                raise ValueError("Historico de validacao inesperado.")

            if loss != "cross_entropy" and (
                accuracy_history or val_accuracy_history
            ):
                raise ValueError("Historico de acuracia inesperado.")

            if validation is None and val_accuracy_history:
                raise ValueError("Historico de acuracia de validacao inesperado.")

            saved_best = saved.get("best_loss")
            best_epoch = saved.get("best_epoch", 0)
            best_state = saved.get("best_state")
            stale = saved.get("stale", 0)

            if saved_best is not None and (
                isinstance(saved_best, bool)
                or not isinstance(saved_best, (int, float))
                or not math.isfinite(saved_best)
                or saved_best < 0
            ):
                raise ValueError("best_loss invalido.")

            if (
                isinstance(best_epoch, bool)
                or not isinstance(best_epoch, int)
                or not 0 <= best_epoch <= start_epoch
            ):
                raise ValueError("best_epoch invalido.")

            if (
                isinstance(stale, bool) or not isinstance(stale, int)
                or not 0 <= stale <= start_epoch
            ):
                raise ValueError("Contador de early stopping invalido.")

            if (saved_best is None) != (best_epoch == 0):
                raise ValueError("Estado de melhor epoca inconsistente.")

            if best_state is not None:
                candidate_best = copy.deepcopy(network)
                _restore(candidate_best, best_state)

            if patience is not None and restore_best and best_epoch > 0:
                if best_state is None:
                    raise ValueError("Checkpoint nao contem o melhor estado da rede.")

            if patience is None and (
                best_epoch != 0 or saved_best is not None
                or best_state is not None or stale != 0
            ):
                raise ValueError("Estado de early stopping inesperado.")

            # Pre-validacao em objetos temporarios: um checkpoint invalido nao
            # pode deixar a rede ou o otimizador parcialmente restaurados.
            network_state = saved.get("network_state")
            candidate_network = copy.deepcopy(network)
            _restore(candidate_network, network_state)

            candidate_optimizer = copy.deepcopy(self.optimizer)
            candidate_optimizer.load_state_dict(
                saved.get("optimizer_state"), network
            )

            candidate_rng = random.Random()
            try:
                candidate_rng.setstate(
                    _rng_from_json(saved.get("rng_state"))
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("Estado aleatorio do checkpoint invalido.") from exc

            # Aplicacao apenas depois de todos os campos terem sido validados.
            _restore(network, network_state)
            self.optimizer.__dict__.clear()
            self.optimizer.__dict__.update(candidate_optimizer.__dict__)
            rng.setstate(candidate_rng.getstate())

            best_loss = math.inf if saved_best is None else saved_best

        def save_progress(epoch):
            if checkpoint_path is None:
                return

            current_best = None if math.isinf(best_loss) else best_loss

            payload = {
                "epoch": epoch,
                "config": config,
                "dataset_signature": dataset_hash,
                "validation_signature": validation_hash,
                "network_shape": shape,
                "network_state": _snapshot(network),
                "optimizer_state": self.optimizer.state_dict(network),
                "history": history,
                "val_history": val_history,
                "accuracy_history": accuracy_history,
                "val_accuracy_history": val_accuracy_history,
                "best_loss": current_best,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "stale": stale,
                "order": order,
                "rng_state": _rng_to_json(rng.getstate()),
            }

            save_training_session(checkpoint_path, payload)

        for epoch in range(start_epoch + 1, epochs + 1):
            if shuffle:
                rng.shuffle(order)

            total_error = 0.0
            hits = 0

            for start in range(0, len(order), batch_size):
                batch = [samples[i] for i in order[start:start + batch_size]]
                batch_gradients = []

                for inputs, targets in batch:
                    result = backpropagate(
                        network, inputs, targets, loss=loss
                    )
                    total_error += loss_fn(result["predictions"], targets)

                    if loss == "cross_entropy":
                        hits += (
                            _argmax(result["predictions"]) == _argmax(targets)
                        )

                    batch_gradients.append(result["gradients"])

                gradients = (
                    batch_gradients[0]
                    if len(batch_gradients) == 1
                    else _average_gradients(batch_gradients)
                )

                if clip_norm is not None:
                    gradients = _clip_gradients(gradients, clip_norm)

                self.optimizer.step(network, gradients)

            train_loss = total_error / len(samples)
            history.append(train_loss)

            if loss == "cross_entropy":
                accuracy_history.append(hits / len(samples))

            val_loss = None

            if validation is not None:
                val_loss = self._loss(network, validation, loss)
                val_history.append(val_loss)

                if loss == "cross_entropy":
                    val_accuracy_history.append(
                        self._accuracy(network, validation)
                    )

            should_stop = False

            if callback is not None and callback(epoch, train_loss, val_loss):
                stopped_early = True
                should_stop = True

            elif target_loss is not None and train_loss <= target_loss:
                stopped_early = True
                should_stop = True

            elif patience is not None:
                monitored = val_loss if val_loss is not None else train_loss

                if monitored < best_loss - min_delta:
                    best_loss = monitored
                    best_epoch = epoch
                    stale = 0

                    if restore_best:
                        best_state = _snapshot(network)
                else:
                    stale += 1

                    if stale >= patience:
                        stopped_early = True
                        should_stop = True

            if checkpoint_path is not None:
                if (
                    epoch % checkpoint_every == 0
                    or epoch == epochs
                    or should_stop
                ):
                    save_progress(epoch)

            if should_stop:
                break

        if patience is not None and restore_best and best_state is not None:
            _restore(network, best_state)

        result = {
            "epochs": epochs,
            "epochs_run": len(history),
            "final_loss": history[-1],
            "history": history,
            "stopped_early": stopped_early,
        }

        if loss == "cross_entropy":
            result["accuracy_history"] = accuracy_history
            result["final_accuracy"] = accuracy_history[-1]

            if validation is not None:
                result["val_accuracy_history"] = val_accuracy_history

        if validation is not None:
            result["val_history"] = val_history
            result["final_val_loss"] = val_history[-1]

        if patience is not None:
            result["best_epoch"] = best_epoch

        return result
