import random

from alpha_math.tensor import Tensor


class Embedding:
    """Converte IDs de tokens em vetores numéricos."""

    def __init__(self, vocab_size, embedding_dim, seed=42):
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
            raise TypeError("O tamanho do vocabulário deve ser inteiro.")

        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int):
            raise TypeError("A dimensão deve ser inteira.")

        if vocab_size <= 0:
            raise ValueError("O vocabulário deve ter pelo menos um token.")

        if embedding_dim <= 0:
            raise ValueError("A dimensão deve ser positiva.")

        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int)
        ):
            raise TypeError("A seed deve ser inteira ou None.")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        generator = random.Random(seed)

        self.weights = Tensor([
            [
                generator.uniform(-0.1, 0.1)
                for _ in range(embedding_dim)
            ]
            for _ in range(vocab_size)
        ])

    def get(self, token_id):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("O ID deve ser inteiro.")

        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError("ID fora dos limites do embedding.")

        return Tensor(self.weights.data[token_id])

    def forward(self, token_ids):
        if not isinstance(token_ids, list):
            raise TypeError("Os IDs devem estar numa lista.")

        vectors = [
            self.get(token_id).data
            for token_id in token_ids
        ]

        return Tensor(vectors)
