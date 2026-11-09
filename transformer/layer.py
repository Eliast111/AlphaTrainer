"""Alpha Transformer Engine
Camada Transformer com Attention, Feed-Forward,
Residual Connections e Layer Normalization.

Suporta forward com cache e backward (retropropagacao) completo.
"""

import math
import random

from alpha_math.tensor import Tensor
from alpha_math.linalg import matmul, transpose, add, column_sums
from transformer.attention import SelfAttention


class TransformerLayer:
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        causal=False,
        epsilon=1e-6,
        seed=None
    ):
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

        if seed is None:
            # Inicializacao deterministica (comportamento original).
            self.w1 = [
                [
                    (((i + 1) * (j + 1)) % 7 - 3) * scale1
                    for j in range(hidden_dim)
                ]
                for i in range(embedding_dim)
            ]

            self.w2 = [
                [
                    (((i + 2) * (j + 1)) % 7 - 3) * scale2
                    for j in range(embedding_dim)
                ]
                for i in range(hidden_dim)
            ]
        else:
            generator = random.Random(seed)
            limit1 = math.sqrt(6.0 / (embedding_dim + hidden_dim))

            self.w1 = [
                [generator.uniform(-limit1, limit1) for _ in range(hidden_dim)]
                for _ in range(embedding_dim)
            ]

            self.w2 = [
                [generator.uniform(-limit1, limit1) for _ in range(embedding_dim)]
                for _ in range(hidden_dim)
            ]

        self.b1 = [0.0] * hidden_dim
        self.b2 = [0.0] * embedding_dim

    # ------------------------------------------------------------------
    # Validacao

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

    # ------------------------------------------------------------------
    # Layer norm (sem parametros aprendidos)

    def _layer_norm_rows(self, rows):
        output = []
        inverse_stds = []

        for row in rows:
            mean = sum(row) / len(row)
            variance = sum((v - mean) ** 2 for v in row) / len(row)
            inverse = 1.0 / math.sqrt(variance + self.epsilon)

            output.append([(v - mean) * inverse for v in row])
            inverse_stds.append(inverse)

        return output, inverse_stds

    @staticmethod
    def _layer_norm_backward(d_output, normalized, inverse_stds):
        result = []

        for dy, y, inverse in zip(d_output, normalized, inverse_stds):
            size = len(dy)
            mean_dy = sum(dy) / size
            mean_dy_y = sum(a * b for a, b in zip(dy, y)) / size

            result.append([
                inverse * (a - mean_dy - b * mean_dy_y)
                for a, b in zip(dy, y)
            ])

        return result

    def _layer_norm(self, inputs):
        output, _ = self._layer_norm_rows(inputs.data)
        return Tensor(output)

    # ------------------------------------------------------------------
    # Feed-forward

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

    # ------------------------------------------------------------------
    # Forward

    def forward_cache(self, rows):
        """Forward sobre lista de listas, guardando o necessario ao backward."""
        x = [list(r) for r in rows]

        attention_out, scores = self.attention.forward(
            Tensor(x), Tensor(x), Tensor(x)
        )

        residual1 = add(x, attention_out.data)
        norm1, inv1 = self._layer_norm_rows(residual1)

        hidden_pre = [
            [a + b for a, b in zip(row, self.b1)]
            for row in matmul(norm1, self.w1)
        ]
        hidden = [[v if v > 0.0 else 0.0 for v in row] for row in hidden_pre]

        ff = [
            [a + b for a, b in zip(row, self.b2)]
            for row in matmul(hidden, self.w2)
        ]

        residual2 = add(norm1, ff)
        output, inv2 = self._layer_norm_rows(residual2)

        cache = {
            "x": x,
            "scores": scores.data,
            "norm1": norm1,
            "inv1": inv1,
            "hidden_pre": hidden_pre,
            "hidden": hidden,
            "output": output,
            "inv2": inv2,
        }

        return output, cache

    def forward(self, inputs):
        self._validate(inputs)
        output, _ = self.forward_cache(inputs.data)
        return Tensor(output)

    def forward_with_attention(self, inputs):
        self._validate(inputs)
        output, cache = self.forward_cache(inputs.data)
        return Tensor(output), Tensor(cache["scores"])

    # ------------------------------------------------------------------
    # Backward

    def parameters(self):
        """Lista (nome, array) dos parametros treinaveis."""
        return [
            ("w1", self.w1), ("b1", self.b1),
            ("w2", self.w2), ("b2", self.b2),
        ]

    def backward(self, d_output, cache):
        """Retropropaga d_output (n x d).

        Devolve (d_entrada, gradientes) onde gradientes e um dict
        com dw1, db1, dw2, db2.
        """
        d_residual2 = self._layer_norm_backward(
            d_output, cache["output"], cache["inv2"]
        )

        # residual2 = norm1 + ff
        d_ff = d_residual2
        hidden = cache["hidden"]

        dw2 = matmul(transpose(hidden), d_ff)
        db2 = column_sums(d_ff)

        d_hidden = matmul(d_ff, transpose(self.w2))
        d_hidden_pre = [
            [g if pre > 0.0 else 0.0 for g, pre in zip(grow, prow)]
            for grow, prow in zip(d_hidden, cache["hidden_pre"])
        ]

        dw1 = matmul(transpose(cache["norm1"]), d_hidden_pre)
        db1 = column_sums(d_hidden_pre)

        d_norm1 = add(d_residual2, matmul(d_hidden_pre, transpose(self.w1)))

        d_residual1 = self._layer_norm_backward(
            d_norm1, cache["norm1"], cache["inv1"]
        )

        # residual1 = x + attention(x, x, x)
        x = cache["x"]
        dq, dk, dv = self.attention.backward(
            x, x, x, cache["scores"], d_residual1
        )

        d_input = add(add(d_residual1, dq), add(dk, dv))

        return d_input, {"w1": dw1, "b1": db1, "w2": dw2, "b2": db2}
