"""
Alpha Learning Engine V0.4
Otimizadores: Gradient Descent, Momentum e Adam.
"""

import math


def _positive_finite(value, name):
    value = float(value)

    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} deve ser positivo e finito.")

    return value


def _unit_interval(value, name):
    value = float(value)

    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(f"{name} deve estar em [0, 1).")

    return value


class _Optimizer:
    """Base: valida os gradientes e aplica uma regra de atualizacao."""

    def __init__(self, learning_rate=0.01):
        self.learning_rate = _positive_finite(
            learning_rate, "Learning rate"
        )

    def _validated_updates(self, network, gradients):
        if len(gradients) != len(network.layers):
            raise ValueError("Numero de camadas incompativel.")

        # Valida todos os gradientes antes de alterar os parametros.
        updates = []

        for layer, layer_gradients in zip(network.layers, gradients):
            if len(layer_gradients) != len(layer.neurons):
                raise ValueError("Numero de neuronios incompativel.")

            for neuron, gradient in zip(layer.neurons, layer_gradients):
                if not isinstance(gradient, dict):
                    raise TypeError("Gradiente invalido.")

                weight_gradients = gradient.get("weights")
                bias_gradient = gradient.get("bias")

                if weight_gradients is None or bias_gradient is None:
                    raise ValueError("Gradientes incompletos.")

                if len(weight_gradients) != len(neuron.weights):
                    raise ValueError("Numero de pesos incompativel.")

                values = [float(value) for value in weight_gradients]
                values.append(float(bias_gradient))

                if not all(math.isfinite(value) for value in values):
                    raise ValueError("Os gradientes devem ser finitos.")

                updates.append((neuron, values))

        return updates

    def step(self, network, gradients):
        updates = self._validated_updates(network, gradients)

        for neuron, values in updates:
            deltas = self._deltas(neuron, values)

            for index in range(len(neuron.weights)):
                neuron.weights[index] -= deltas[index]

            neuron.bias -= deltas[-1]

    def _deltas(self, neuron, values):
        raise NotImplementedError


class GradientDescent(_Optimizer):
    """Descida do gradiente simples (SGD)."""

    def _deltas(self, neuron, values):
        return [self.learning_rate * value for value in values]


class Momentum(_Optimizer):
    """SGD com momento (acelera e suaviza a descida)."""

    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = _unit_interval(momentum, "Momentum")
        self._velocity = {}

    def _deltas(self, neuron, values):
        velocity = self._velocity.setdefault(
            id(neuron), [0.0] * len(values)
        )

        for index, value in enumerate(values):
            velocity[index] = (
                self.momentum * velocity[index]
                + self.learning_rate * value
            )

        return list(velocity)


class Adam(_Optimizer):
    """Adam: taxa de aprendizagem adaptativa por parametro."""

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):
        super().__init__(learning_rate)
        self.beta1 = _unit_interval(beta1, "beta1")
        self.beta2 = _unit_interval(beta2, "beta2")
        self.epsilon = _positive_finite(epsilon, "epsilon")
        self._state = {}

    def _deltas(self, neuron, values):
        state = self._state.setdefault(
            id(neuron),
            {"t": 0, "m": [0.0] * len(values), "v": [0.0] * len(values)}
        )

        state["t"] += 1
        t = state["t"]
        correction1 = 1.0 - self.beta1 ** t
        correction2 = 1.0 - self.beta2 ** t

        deltas = []

        for index, gradient in enumerate(values):
            m = self.beta1 * state["m"][index] + (1 - self.beta1) * gradient
            v = (
                self.beta2 * state["v"][index]
                + (1 - self.beta2) * gradient * gradient
            )

            state["m"][index] = m
            state["v"][index] = v

            deltas.append(
                self.learning_rate * (m / correction1)
                / (math.sqrt(v / correction2) + self.epsilon)
            )

        return deltas


OPTIMIZERS = {
    "sgd": GradientDescent,
    "momentum": Momentum,
    "adam": Adam,
}


def create_optimizer(name="sgd", learning_rate=None, **kwargs):
    name = str(name).lower()

    if name not in OPTIMIZERS:
        raise ValueError(
            "Otimizador desconhecido. Disponiveis: "
            + ", ".join(OPTIMIZERS)
        )

    if learning_rate is None:
        return OPTIMIZERS[name](**kwargs)

    return OPTIMIZERS[name](learning_rate=learning_rate, **kwargs)
