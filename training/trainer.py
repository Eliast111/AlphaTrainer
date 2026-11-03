"""
Alpha Training Engine V0.1
Treinamento automatico de redes neurais.
"""

from learning.backprop import backpropagate
from learning.loss import mse


class Trainer:
    def __init__(self, optimizer):
        if not callable(getattr(optimizer, "step", None)):
            raise TypeError("Optimizer invalido.")

        self.optimizer = optimizer

    def fit(self, network, dataset, epochs=100):
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
            raise ValueError("Epochs deve ser um inteiro positivo.")

        if not dataset:
            raise ValueError("O dataset nao pode estar vazio.")

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

            mse(prediction, targets)
            samples.append((inputs, targets))

        history = []

        for epoch in range(epochs):
            total_error = 0.0

            for inputs, targets in samples:
                prediction = network.forward(inputs)
                total_error += mse(prediction, targets)

                result = backpropagate(network, inputs, targets)

                self.optimizer.step(
                    network,
                    result["gradients"]
                )

            average_error = total_error / len(samples)
            history.append(average_error)

        return {
            "epochs": epochs,
            "final_loss": history[-1],
            "history": history
        }
