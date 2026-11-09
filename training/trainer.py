"""
Alpha Training Engine V0.2
Treinamento de redes neuronais: mini-batches, shuffle, validacao,
early stopping, gradient clipping e callbacks.
"""

import math
import random

from learning.backprop import backpropagate
from learning.loss import mse, cross_entropy, softmax


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

            # Verifica a amostra antes de iniciar as atualizacoes.
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
        loss="mse"
    ):
        """Treina a rede e devolve o historico.

        batch_size: 1 = SGD por amostra; N = mini-batch; len(dataset) = batch.
        validation_data: dataset de validacao (opcional).
        patience: epocas sem melhoria (na validacao, ou treino se nao houver
            validacao) antes de parar. None desativa o early stopping.
        restore_best: com early stopping, repoe os melhores pesos.
        clip_norm: limita a norma global do gradiente.
        target_loss: pára quando a perda de treino for <= a este valor.
        callback: funcao(epoca, perda_treino, perda_validacao) -> True para parar.
        loss: "mse" (regressao) ou "cross_entropy" (classificacao com alvos
            one-hot; a camada de saida deve ser linear). Em classificacao o
            resultado inclui o historico de acuracia.
        """
        loss_fn = _loss_function(loss)
        if not _is_positive_int(epochs):
            raise ValueError("Epochs deve ser um inteiro positivo.")

        if not _is_positive_int(batch_size):
            raise ValueError("batch_size deve ser um inteiro positivo.")

        if patience is not None and not _is_positive_int(patience):
            raise ValueError("patience deve ser um inteiro positivo ou None.")

        if clip_norm is not None and not (
            isinstance(clip_norm, (int, float))
            and not isinstance(clip_norm, bool)
            and math.isfinite(clip_norm)
            and clip_norm > 0
        ):
            raise ValueError("clip_norm deve ser positivo.")

        if callback is not None and not callable(callback):
            raise TypeError("callback deve ser uma funcao.")

        samples = self._prepare(network, dataset, loss=loss)

        validation = None

        if validation_data is not None:
            validation = self._prepare(
                network, validation_data, "dataset de validacao", loss=loss
            )

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

        for epoch in range(1, epochs + 1):
            if shuffle:
                rng.shuffle(order)

            total_error = 0.0
            hits = 0

            for start in range(0, len(order), batch_size):
                batch = [samples[i] for i in order[start:start + batch_size]]
                batch_gradients = []

                for inputs, targets in batch:
                    result = backpropagate(network, inputs, targets, loss=loss)
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

            if callback is not None and callback(epoch, train_loss, val_loss):
                stopped_early = True
                break

            if target_loss is not None and train_loss <= target_loss:
                stopped_early = True
                break

            if patience is not None:
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


def _snapshot(network):
    return [
        [(list(n.weights), n.bias) for n in layer.neurons]
        for layer in network.layers
    ]


def _restore(network, state):
    for layer, layer_state in zip(network.layers, state):
        for neuron, (weights, bias) in zip(layer.neurons, layer_state):
            neuron.weights = list(weights)
            neuron.bias = bias
