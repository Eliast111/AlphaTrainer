"""
Alpha Neural Engine V0.2
Rede neural com suporte a cache de treinamento.
"""

from neural.layer import Layer


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
