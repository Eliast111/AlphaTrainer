"""Gradientes da camada de saida de modelos de linguagem."""


def output_layer_gradients(hidden_states, probabilities, targets, output_weights):
    if not all(isinstance(value, (list, tuple)) for value in (
        hidden_states, probabilities, targets, output_weights
    )):
        raise TypeError("Os argumentos devem ser listas ou tuplas.")

    if not hidden_states or len(hidden_states) != len(probabilities):
        raise ValueError("Estados ocultos e probabilidades devem ter o mesmo tamanho.")

    if len(targets) != len(hidden_states):
        raise ValueError("Deve existir um alvo para cada posicao.")

    if not output_weights or not output_weights[0]:
        raise ValueError("Os pesos de saida nao podem estar vazios.")

    embedding_dim = len(output_weights)
    vocab_size = len(output_weights[0])

    if any(len(row) != vocab_size for row in output_weights):
        raise ValueError("A matriz de pesos deve ser retangular.")

    if any(len(row) != embedding_dim for row in hidden_states):
        raise ValueError("Dimensao dos estados ocultos incorreta.")

    if any(len(row) != vocab_size for row in probabilities):
        raise ValueError("Dimensao das probabilidades incorreta.")

    for target in targets:
        if isinstance(target, bool) or not isinstance(target, int):
            raise TypeError("Cada alvo deve ser inteiro.")
        if target < 0 or target >= vocab_size:
            raise ValueError("Alvo fora do vocabulario.")

    count = len(targets)
    weight_gradients = [[0.0] * vocab_size for _ in range(embedding_dim)]
    bias_gradients = [0.0] * vocab_size
    hidden_gradients = []

    for hidden, row, target in zip(hidden_states, probabilities, targets):
        if any(not 0.0 <= float(value) <= 1.0 for value in row):
            raise ValueError("Probabilidades fora do intervalo valido.")

        logit_gradients = [
            (float(probability) - (1.0 if index == target else 0.0)) / count
            for index, probability in enumerate(row)
        ]

        for i in range(embedding_dim):
            for j in range(vocab_size):
                weight_gradients[i][j] += float(hidden[i]) * logit_gradients[j]

        for j in range(vocab_size):
            bias_gradients[j] += logit_gradients[j]

        hidden_gradients.append([
            sum(logit_gradients[j] * float(output_weights[i][j])
                for j in range(vocab_size))
            for i in range(embedding_dim)
        ])

    return {
        "weights": weight_gradients,
        "bias": bias_gradients,
        "hidden": hidden_gradients,
    }
