"""Alpha Transformer Engine
Camada Transformer com Attention, Feed-Forward,
Residual Connections e Layer Normalization.
"""

import math

from alpha_math.tensor import Tensor
from transformer.attention import SelfAttention


class TransformerLayer:
    def __init__(self, embedding_dim, hidden_dim, causal=False, epsilon=1e-6):
        for name, value in (
            ("embedding_dim", embedding_dim),
            ("hidden_dim", hidden_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} deve ser inteiro.")
            if value <= 0:
                raise ValueError(f"{name} deve ser positivo.")

        if not isinstance(causal, bool):
            raise TypeError("causal deve ser booleano.")

        if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool):
            raise TypeError("epsilon deve ser numérico.")

        if epsilon <= 0:
            raise ValueError("epsilon deve ser positivo.")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.epsilon = float(epsilon)

        self.attention = SelfAttention(causal=causal)

        scale1 = 1.0 / math.sqrt(embedding_dim)
        scale2 = 1.0 / math.sqrt(hidden_dim)

        self.w1 = [
            [
                (((i + 1) * (j + 1)) % 7 - 3) * scale1
                for j in range(hidden_dim)
            ]
            for i in range(embedding_dim)
        ]

        self.b1 = [0.0] * hidden_dim

        self.w2 = [
            [
                (((i + 2) * (j + 1)) % 7 - 3) * scale2
                for j in range(embedding_dim)
            ]
            for i in range(hidden_dim)
        ]

        self.b2 = [0.0] * embedding_dim

    def _validate(self, inputs):
        if not isinstance(inputs, Tensor):
            raise TypeError("As entradas devem ser Tensor.")

        shape = inputs.shape()

        if len(shape) != 2:
            raise ValueError("As entradas devem ser uma matriz.")

        rows, cols = shape

        if rows == 0:
            raise ValueError("A sequência não pode estar vazia.")

        if cols != self.embedding_dim:
            raise ValueError("Dimensão dos embeddings incompatível.")

    def _layer_norm(self, inputs):
        output = []

        for row in inputs.data:
            mean = sum(row) / len(row)

            variance = sum(
                (value - mean) ** 2 for value in row
            ) / len(row)

            denominator = math.sqrt(variance + self.epsilon)

            output.append([
                (value - mean) / denominator
                for value in row
            ])

        return Tensor(output)

    def _feed_forward(self, inputs):
        output = []

        for row in inputs.data:
            hidden = []

            for j in range(self.hidden_dim):
                value = sum(
                    row[i] * self.w1[i][j]
                    for i in range(self.embedding_dim)
                ) + self.b1[j]

                hidden.append(max(0.0, value))

            result = []

            for j in range(self.embedding_dim):
                value = sum(
                    hidden[i] * self.w2[i][j]
                    for i in range(self.hidden_dim)
                ) + self.b2[j]

                result.append(value)

            output.append(result)

        return Tensor(output)

    def forward(self, inputs):
        self._validate(inputs)

        attention_output, scores = self.attention.forward(
            inputs, inputs, inputs
        )

        attention_residual = inputs.add(attention_output)
        normalized_attention = self._layer_norm(attention_residual)

        feed_forward_output = self._feed_forward(normalized_attention)

        final_residual = normalized_attention.add(feed_forward_output)
        output = self._layer_norm(final_residual)

        return output

    def forward_with_attention(self, inputs):
        self._validate(inputs)

        attention_output, scores = self.attention.forward(
            inputs, inputs, inputs
        )

        attention_residual = inputs.add(attention_output)
        normalized_attention = self._layer_norm(attention_residual)

        feed_forward_output = self._feed_forward(normalized_attention)

        final_residual = normalized_attention.add(feed_forward_output)
        output = self._layer_norm(final_residual)

        return output, scores
