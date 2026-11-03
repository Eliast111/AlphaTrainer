import math

from alpha_math.tensor import Tensor


class SelfAttention:
    """Implementação inicial de Scaled Dot-Product Attention."""

    def __init__(self, causal=False):
        if not isinstance(causal, bool):
            raise TypeError("causal deve ser booleano.")

        self.causal = causal

    def softmax(self, values):
        if not values:
            raise ValueError("A lista não pode estar vazia.")

        maximum = max(values)
        exponents = [math.exp(x - maximum) for x in values]
        total = sum(exponents)

        return [x / total for x in exponents]

    def forward(self, query, key, value):
        if not all(isinstance(x, Tensor) for x in (query, key, value)):
            raise TypeError("Query, Key e Value devem ser Tensors.")

        if len(query.shape()) != 2:
            raise ValueError("Query deve ser uma matriz.")

        if len(key.shape()) != 2 or len(value.shape()) != 2:
            raise ValueError("Key e Value devem ser matrizes.")

        q_rows, q_dim = query.shape()
        k_rows, k_dim = key.shape()
        v_rows, v_dim = value.shape()

        if q_rows == 0 or k_rows == 0 or v_rows == 0:
            raise ValueError("As matrizes não podem estar vazias.")

        if q_dim != k_dim:
            raise ValueError("Dimensões de Query e Key incompatíveis.")

        if k_rows != v_rows:
            raise ValueError("Key e Value devem ter o mesmo número de linhas.")

        scores = []

        for i in range(q_rows):
            row = []

            for j in range(k_rows):
                if self.causal and j > i:
                    row.append(float("-inf"))
                    continue

                score = sum(
                    query.data[i][d] * key.data[j][d]
                    for d in range(q_dim)
                )

                row.append(score / math.sqrt(q_dim))

            scores.append(self.softmax(row))

        output = []

        for i in range(q_rows):
            row = []

            for d in range(v_dim):
                result = sum(
                    scores[i][j] * value.data[j][d]
                    for j in range(k_rows)
                )
                row.append(result)

            output.append(row)

        return Tensor(output), Tensor(scores)
