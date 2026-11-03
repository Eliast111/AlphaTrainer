import math

from alpha_math.tensor import Tensor


class PositionalEncoding:
    """Adiciona informação de posição aos embeddings."""

    def __init__(self, max_length, embedding_dim):
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise TypeError("max_length deve ser inteiro.")

        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int):
            raise TypeError("embedding_dim deve ser inteiro.")

        if max_length <= 0:
            raise ValueError("max_length deve ser positivo.")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim deve ser positivo.")

        self.max_length = max_length
        self.embedding_dim = embedding_dim

        positions = []

        for pos in range(max_length):
            row = []

            for i in range(embedding_dim):
                exponent = (2 * (i // 2)) / embedding_dim
                angle = pos / (10000 ** exponent)

                if i % 2 == 0:
                    row.append(math.sin(angle))
                else:
                    row.append(math.cos(angle))

            positions.append(row)

        self.encoding = Tensor(positions)

    def get(self, position):
        if isinstance(position, bool) or not isinstance(position, int):
            raise TypeError("A posição deve ser inteira.")

        if position < 0 or position >= self.max_length:
            raise ValueError("Posição fora dos limites.")

        return Tensor(self.encoding.data[position])

    def forward(self, embeddings):
        if not isinstance(embeddings, Tensor):
            raise TypeError("Os embeddings devem ser Tensor.")

        shape = embeddings.shape()

        if len(shape) != 2:
            raise ValueError("Os embeddings devem ser uma matriz.")

        sequence_length, embedding_dim = shape

        if sequence_length == 0:
            raise ValueError("A sequência não pode estar vazia.")

        if sequence_length > self.max_length:
            raise ValueError("Sequência maior que max_length.")

        if embedding_dim != self.embedding_dim:
            raise ValueError("Dimensão dos embeddings incompatível.")

        position_matrix = Tensor(
            self.encoding.data[:sequence_length]
        )

        return embeddings.add(position_matrix)
