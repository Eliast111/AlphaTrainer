"""Gradientes dos embeddings de modelos de linguagem."""


def embedding_gradients(token_ids, hidden_gradients, vocab_size, embedding_dim):
    if not isinstance(token_ids, (list, tuple)):
        raise TypeError("Os IDs devem estar numa lista.")
    if not isinstance(hidden_gradients, (list, tuple)):
        raise TypeError("Os gradientes devem estar numa lista.")
    if not token_ids or len(token_ids) != len(hidden_gradients):
        raise ValueError("IDs e gradientes devem ter o mesmo tamanho.")

    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError("vocab_size deve ser inteiro.")
    if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int):
        raise TypeError("embedding_dim deve ser inteiro.")
    if vocab_size <= 0 or embedding_dim <= 0:
        raise ValueError("As dimensoes devem ser positivas.")

    gradients = [[0.0] * embedding_dim for _ in range(vocab_size)]

    for token_id, gradient in zip(token_ids, hidden_gradients):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("Cada ID deve ser inteiro.")
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError("ID fora do vocabulario.")
        if len(gradient) != embedding_dim:
            raise ValueError("Dimensao do gradiente incorreta.")

        for index, value in enumerate(gradient):
            gradients[token_id][index] += float(value)

    return gradients
