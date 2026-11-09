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

    def backward(self, query, key, value, scores, d_output):
        """Retropropagacao da atencao.

        query/key/value/scores/d_output: listas de listas.
        scores sao os pesos (apos softmax) devolvidos pelo forward.
        Devolve (dQuery, dKey, dValue).
        """
        q_rows = len(query)
        k_rows = len(key)
        q_dim = len(query[0])
        v_dim = len(value[0])
        scale = 1.0 / math.sqrt(q_dim)

        d_value = [[0.0] * v_dim for _ in range(k_rows)]
        d_query = [[0.0] * q_dim for _ in range(q_rows)]
        d_key = [[0.0] * q_dim for _ in range(k_rows)]

        for i in range(q_rows):
            weights = scores[i]
            grad_out = d_output[i]

            # dA[i][j] = dO[i] . V[j]
            d_weights = [
                sum(g * v for g, v in zip(grad_out, value[j]))
                for j in range(k_rows)
            ]

            # dV[j] += A[i][j] * dO[i]
            for j in range(k_rows):
                a = weights[j]

                if a != 0.0:
                    row = d_value[j]

                    for d in range(v_dim):
                        row[d] += a * grad_out[d]

            # Softmax: dS = A * (dA - sum(dA * A))
            inner = sum(a * g for a, g in zip(weights, d_weights))

            for j in range(k_rows):
                a = weights[j]

                if a == 0.0:
                    continue

                d_score = a * (d_weights[j] - inner) * scale

                for d in range(q_dim):
                    d_query[i][d] += d_score * key[j][d]
                    d_key[j][d] += d_score * query[i][d]

        return d_query, d_key, d_value
