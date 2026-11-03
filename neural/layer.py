"""
Alpha Neural Engine V0.2
Camada com suporte a cache de treinamento.
"""

from neural.neuron import Neuron


class Layer:
    def __init__(self, neurons):
        if not neurons:
            raise ValueError("A camada precisa de pelo menos um neuronio.")

        if not all(isinstance(n, Neuron) for n in neurons):
            raise TypeError("Todos os elementos devem ser neuronios.")

        self.neurons = list(neurons)

    def forward_with_cache(self, inputs):
        outputs = []
        caches = []

        for neuron in self.neurons:
            output, z = neuron.forward_with_cache(inputs)
            outputs.append(output)
            caches.append({
                "z": z,
                "output": output
            })

        return outputs, caches

    def forward(self, inputs):
        outputs, _ = self.forward_with_cache(inputs)
        return outputs

    def size(self):
        return len(self.neurons)
