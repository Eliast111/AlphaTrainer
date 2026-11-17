
"""
Alpha Learning Engine V0.5
Otimizadores com suporte a guardar e restaurar estado.
"""

import math


STATE_FORMAT = "alpha_optimizer_state_v1"


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


def _finite_list(values, name):
    if not isinstance(values, list):
        raise ValueError(f"{name} deve ser uma lista.")

    result = [float(value) for value in values]

    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} deve conter apenas valores finitos.")

    return result


def _network_neurons(network):
    return [
        neuron
        for layer in network.layers
        for neuron in layer.neurons
    ]


def _network_signature(network):
    return [
        [len(neuron.weights) for neuron in layer.neurons]
        for layer in network.layers
    ]


class _Optimizer:
    """Base: valida gradientes e aplica atualizações."""

    optimizer_name = None

    def __init__(self, learning_rate=0.01):
        self.learning_rate = _positive_finite(
            learning_rate, "Learning rate"
        )

    def _validated_updates(self, network, gradients):
        if len(gradients) != len(network.layers):
            raise ValueError("Numero de camadas incompativel.")

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

    def _base_state(self, network):
        return {
            "format": STATE_FORMAT,
            "optimizer": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "network_signature": _network_signature(network),
        }

    def state_dict(self, network):
        """Exporta o estado do otimizador em formato serializável."""
        return self._base_state(network)

    def _validate_state_header(self, state, network):
        if not isinstance(state, dict):
            raise TypeError("Estado do otimizador invalido.")

        if state.get("format") != STATE_FORMAT:
            raise ValueError("Formato de estado desconhecido.")

        if state.get("optimizer") != self.optimizer_name:
            raise ValueError("Tipo de otimizador incompativel.")

        if state.get("network_signature") != _network_signature(network):
            raise ValueError("Arquitetura da rede incompativel.")

        learning_rate = _positive_finite(
            state.get("learning_rate"), "Learning rate"
        )

        return learning_rate

    def load_state_dict(self, state, network):
        """Valida e restaura a configuração do otimizador."""
        learning_rate = self._validate_state_header(state, network)
        self.learning_rate = learning_rate


class GradientDescent(_Optimizer):
    """Descida do gradiente simples (SGD)."""

    optimizer_name = "sgd"

    def _deltas(self, neuron, values):
        return [self.learning_rate * value for value in values]


class Momentum(_Optimizer):
    """SGD com momento."""

    optimizer_name = "momentum"

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

    def state_dict(self, network):
        state = self._base_state(network)
        state["momentum"] = self.momentum
        state["velocity"] = []

        for neuron in _network_neurons(network):
            velocity = self._velocity.get(id(neuron))
            state["velocity"].append(
                None if velocity is None else list(velocity)
            )

        return state

    def load_state_dict(self, state, network):
        learning_rate = self._validate_state_header(state, network)
        momentum = _unit_interval(state.get("momentum"), "Momentum")

        saved_velocity = state.get("velocity")
        neurons = _network_neurons(network)

        if not isinstance(saved_velocity, list):
            raise ValueError("Estado de Momentum invalido.")

        if len(saved_velocity) != len(neurons):
            raise ValueError("Numero de neuronios incompativel.")

        restored = {}

        for neuron, velocity in zip(neurons, saved_velocity):
            if velocity is None:
                continue

            values = _finite_list(velocity, "Velocity")

            if len(values) != len(neuron.weights) + 1:
                raise ValueError("Dimensao de velocity incompativel.")

            restored[id(neuron)] = values

        self.learning_rate = learning_rate
        self.momentum = momentum
        self._velocity = restored


class Adam(_Optimizer):
    """Adam: taxa de aprendizagem adaptativa por parametro."""

    optimizer_name = "adam"

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
            {
                "t": 0,
                "m": [0.0] * len(values),
                "v": [0.0] * len(values)
            }
        )

        state["t"] += 1
        t = state["t"]
        correction1 = 1.0 - self.beta1 ** t
        correction2 = 1.0 - self.beta2 ** t

        deltas = []

        for index, gradient in enumerate(values):
            m = (
                self.beta1 * state["m"][index]
                + (1 - self.beta1) * gradient
            )
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

    def state_dict(self, network):
        state = self._base_state(network)
        state.update({
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "neurons": []
        })

        for neuron in _network_neurons(network):
            item = self._state.get(id(neuron))

            state["neurons"].append(
                None if item is None else {
                    "t": item["t"],
                    "m": list(item["m"]),
                    "v": list(item["v"])
                }
            )

        return state

    def load_state_dict(self, state, network):
        learning_rate = self._validate_state_header(state, network)
        beta1 = _unit_interval(state.get("beta1"), "beta1")
        beta2 = _unit_interval(state.get("beta2"), "beta2")
        epsilon = _positive_finite(state.get("epsilon"), "epsilon")

        saved_neurons = state.get("neurons")
        neurons = _network_neurons(network)

        if not isinstance(saved_neurons, list):
            raise ValueError("Estado de Adam invalido.")

        if len(saved_neurons) != len(neurons):
            raise ValueError("Numero de neuronios incompativel.")

        restored = {}

        for neuron, item in zip(neurons, saved_neurons):
            if item is None:
                continue

            if not isinstance(item, dict):
                raise ValueError("Estado de neuronio invalido.")

            t = item.get("t")

            if isinstance(t, bool) or not isinstance(t, int) or t < 1:
                raise ValueError("Contador Adam invalido.")

            m = _finite_list(item.get("m"), "Adam m")
            v = _finite_list(item.get("v"), "Adam v")
            expected = len(neuron.weights) + 1

            if len(m) != expected or len(v) != expected:
                raise ValueError("Dimensao do estado Adam incompativel.")

            if any(value < 0 for value in v):
                raise ValueError("Os valores Adam v nao podem ser negativos.")

            restored[id(neuron)] = {
                "t": t,
                "m": m,
                "v": v
            }

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._state = restored


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
