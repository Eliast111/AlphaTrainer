"""
Alpha Training Engine - Language Model
Treino do modelo de linguagem com Adam, mini-batches e clipping.
"""

import math
import random


def _is_positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _zeros_like(array):
    if array and isinstance(array[0], list):
        return [[0.0] * len(row) for row in array]

    return [0.0] * len(array)


class LanguageModelTrainer:
    def __init__(
        self,
        model,
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):
        if not callable(getattr(model, "loss_and_gradients", None)):
            raise TypeError("Modelo de linguagem invalido.")

        learning_rate = float(learning_rate)

        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate deve ser positivo e finito.")

        self.model = model
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.step_count = 0

        self._m = {name: _zeros_like(a) for name, a in model.parameters()}
        self._v = {name: _zeros_like(a) for name, a in model.parameters()}

    # ------------------------------------------------------------------

    @staticmethod
    def _accumulate(total, grads):
        for name, grad in grads.items():
            target = total[name]

            if grad and isinstance(grad[0], list):
                for t_row, g_row in zip(target, grad):
                    for i, value in enumerate(g_row):
                        t_row[i] += value
            else:
                for i, value in enumerate(grad):
                    target[i] += value

    @staticmethod
    def _scale(grads, factor):
        for grad in grads.values():
            if grad and isinstance(grad[0], list):
                for row in grad:
                    for i in range(len(row)):
                        row[i] *= factor
            else:
                for i in range(len(grad)):
                    grad[i] *= factor

    @staticmethod
    def _norm(grads):
        total = 0.0

        for grad in grads.values():
            if grad and isinstance(grad[0], list):
                total += sum(v * v for row in grad for v in row)
            else:
                total += sum(v * v for v in grad)

        return math.sqrt(total)

    def _apply(self, grads):
        self.step_count += 1
        t = self.step_count
        c1 = 1.0 - self.beta1 ** t
        c2 = 1.0 - self.beta2 ** t

        for name, param in self.model.parameters():
            grad = grads[name]
            m = self._m[name]
            v = self._v[name]

            if param and isinstance(param[0], list):
                pairs = zip(param, grad, m, v)
            else:
                pairs = [(param, grad, m, v)]

            for p_row, g_row, m_row, v_row in pairs:
                for i, g in enumerate(g_row):
                    m_row[i] = self.beta1 * m_row[i] + (1 - self.beta1) * g
                    v_row[i] = (
                        self.beta2 * v_row[i] + (1 - self.beta2) * g * g
                    )
                    p_row[i] -= (
                        self.learning_rate * (m_row[i] / c1)
                        / (math.sqrt(v_row[i] / c2) + self.epsilon)
                    )

    # ------------------------------------------------------------------

    def fit(
        self,
        texts,
        epochs=50,
        batch_size=4,
        shuffle=True,
        seed=0,
        clip_norm=1.0,
        callback=None
    ):
        """Treina o modelo. Devolve historico de perda e perplexidade.

        callback(epoca, perda) -> True para parar mais cedo.
        """
        if not _is_positive_int(epochs):
            raise ValueError("Epochs deve ser um inteiro positivo.")

        if not _is_positive_int(batch_size):
            raise ValueError("batch_size deve ser um inteiro positivo.")

        if clip_norm is not None and not clip_norm > 0:
            raise ValueError("clip_norm deve ser positivo.")

        pairs = self.model.make_training_pairs(texts)
        rng = random.Random(seed)
        order = list(range(len(pairs)))
        history = []
        stopped_early = False

        for epoch in range(1, epochs + 1):
            if shuffle:
                rng.shuffle(order)

            epoch_loss = 0.0

            for start in range(0, len(order), batch_size):
                batch = [pairs[i] for i in order[start:start + batch_size]]
                total = {
                    name: _zeros_like(a)
                    for name, a in self.model.parameters()
                }

                for inputs, targets in batch:
                    loss, grads = self.model.loss_and_gradients(
                        inputs, targets
                    )
                    epoch_loss += loss
                    self._accumulate(total, grads)

                self._scale(total, 1.0 / len(batch))

                if clip_norm is not None:
                    norm = self._norm(total)

                    if norm > clip_norm:
                        self._scale(total, clip_norm / norm)

                self._apply(total)

            average = epoch_loss / len(pairs)
            history.append(average)

            if callback is not None and callback(epoch, average):
                stopped_early = True
                break

        return {
            "epochs": epochs,
            "epochs_run": len(history),
            "final_loss": history[-1],
            "perplexity": math.exp(min(history[-1], 700.0)),
            "history": history,
            "stopped_early": stopped_early,
        }
