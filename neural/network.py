"""
Alpha Neural Engine V0.2
Rede neural com suporte a cache de treinamento.
"""

from neural.layer import Layer
from learning.loss import softmax


class Network:
    def __init__(self, layers):
        if not layers:
            raise ValueError("A rede precisa de pelo menos uma camada.")

        if not all(isinstance(layer, Layer) for layer in layers):
            raise TypeError("Todos os elementos devem ser camadas.")

        self.layers = list(layers)

    def forward_with_cache(self, inputs):
        output = list(inputs)
        caches = []

        for layer in self.layers:
            layer_inputs = list(output)
            output, neuron_caches = layer.forward_with_cache(output)

            caches.append({
                "inputs": layer_inputs,
                "neurons": neuron_caches
            })

        return output, caches

    def forward(self, inputs):
        output, _ = self.forward_with_cache(inputs)
        return output

    def layer_count(self):
        return len(self.layers)

    def input_size(self):
        return len(self.layers[0].neurons[0].weights)

    def output_size(self):
        return len(self.layers[-1].neurons)

    def layer_sizes(self):
        """Devolve [entradas, camada1, camada2, ...]."""
        return [self.input_size()] + [layer.size() for layer in self.layers]

    def parameter_count(self):
        return sum(
            len(neuron.weights) + 1
            for layer in self.layers
            for neuron in layer.neurons
        )

    def predict(self, inputs):
        return self.forward(inputs)

    def predict_proba(self, inputs):
        """Probabilidades por classe (softmax das saidas)."""
        return softmax(self.forward(inputs))

    def predict_class(self, inputs):
        """Indice da classe mais provavel."""
        probabilities = self.predict_proba(inputs)
        return max(range(len(probabilities)), key=probabilities.__getitem__)
