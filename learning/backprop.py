"""
Alpha Learning Engine V0.2
Motor de retropropagacao de erros (Backpropagation).
"""

from learning.loss import mse_derivative, softmax


def backpropagate(network, inputs, targets, loss="mse"):
    """loss: "mse" ou "cross_entropy" (classificacao).

    Com cross_entropy a camada de saida deve ser linear: as saidas sao
    tratadas como logits e as predicoes devolvidas sao probabilidades.
    """
    if loss not in ("mse", "cross_entropy"):
        raise ValueError("loss deve ser 'mse' ou 'cross_entropy'.")

    predictions, caches = network.forward_with_cache(inputs)

    if len(predictions) != len(targets):
        raise ValueError("Predicoes e alvos devem ter o mesmo tamanho.")

    if not targets:
        raise ValueError("Os alvos nao podem estar vazios.")

    deltas = [None] * len(network.layers)

    # Calcula o erro na camada de saida.
    output_layer = network.layers[-1]
    output_cache = caches[-1]["neurons"]

    if loss == "cross_entropy":
        if any(n.activation_name != "linear" for n in output_layer.neurons):
            raise ValueError(
                "cross_entropy exige a camada de saida com ativacao linear."
            )

        predictions = softmax(predictions)
        deltas[-1] = [p - float(t) for p, t in zip(predictions, targets)]
    else:
        output_errors = mse_derivative(predictions, targets)

        deltas[-1] = [
            error * neuron.activation_derivative(cache["z"])
            for neuron, cache, error in zip(
                output_layer.neurons,
                output_cache,
                output_errors
            )
        ]

    # Propaga o erro pelas camadas anteriores.
    for layer_index in range(len(network.layers) - 2, -1, -1):
        current_layer = network.layers[layer_index]
        next_layer = network.layers[layer_index + 1]

        current_cache = caches[layer_index]["neurons"]
        next_deltas = deltas[layer_index + 1]

        current_deltas = []

        for neuron_index, (neuron, cache) in enumerate(
            zip(current_layer.neurons, current_cache)
        ):
            propagated_error = sum(
                next_deltas[next_index] *
                next_layer.neurons[next_index].weights[neuron_index]
                for next_index in range(len(next_layer.neurons))
            )

            delta = (
                propagated_error *
                neuron.activation_derivative(cache["z"])
            )

            current_deltas.append(delta)

        deltas[layer_index] = current_deltas

    # Calcula os gradientes dos pesos e dos biases.
    gradients = []

    for layer_index, layer in enumerate(network.layers):
        layer_inputs = caches[layer_index]["inputs"]
        layer_gradients = []

        for neuron, delta in zip(layer.neurons, deltas[layer_index]):
            weight_gradients = [
                delta * float(value)
                for value in layer_inputs
            ]

            layer_gradients.append({
                "weights": weight_gradients,
                "bias": delta
            })

        gradients.append(layer_gradients)

    return {
        "predictions": predictions,
        "gradients": gradients
    }
