"""
Alpha Tensor Engine V0.1
Operacoes matematicas para redes neurais.
"""

class Tensor:
    def __init__(self, data):
        if not isinstance(data, (list, tuple)):
            raise TypeError("Os dados devem ser uma lista ou tupla.")

        self.data = list(data)
        self.shape()

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Tensor({self.data})"

    def shape(self):
        if not self.data:
            return (0,)

        if isinstance(self.data[0], (list, tuple)):
            rows = len(self.data)
            cols = len(self.data[0])

            if any(
                not isinstance(row, (list, tuple)) or len(row) != cols
                for row in self.data
            ):
                raise ValueError("Matriz irregular.")

            if any(
                isinstance(value, (list, tuple))
                for row in self.data
                for value in row
            ):
                raise ValueError("Apenas vetores e matrizes sao suportados.")

            return (rows, cols)

        if any(isinstance(x, (list, tuple)) for x in self.data):
            raise ValueError("Estrutura de Tensor invalida.")

        return (len(self.data),)

    def add(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("O outro objeto deve ser Tensor.")

        if self.shape() != other.shape():
            raise ValueError("Dimensoes incompativeis.")

        if len(self.shape()) == 1:
            return Tensor([a + b for a, b in zip(self.data, other.data)])

        return Tensor([
            [a + b for a, b in zip(ra, rb)]
            for ra, rb in zip(self.data, other.data)
        ])

    def multiply(self, scalar):
        if len(self.shape()) == 1:
            return Tensor([x * scalar for x in self.data])

        return Tensor([
            [x * scalar for x in row]
            for row in self.data
        ])

    def dot(self, other):
        if len(self.shape()) != 1 or len(other.shape()) != 1:
            raise ValueError("Produto escalar exige vetores.")

        if len(self.data) != len(other.data):
            raise ValueError("Dimensoes incompativeis.")

        return sum(a * b for a, b in zip(self.data, other.data))

    def matmul(self, other):
        if len(self.shape()) != 2 or len(other.shape()) != 2:
            raise ValueError("Multiplicacao exige matrizes.")

        rows_a, cols_a = self.shape()
        rows_b, cols_b = other.shape()

        if cols_a != rows_b:
            raise ValueError("Dimensoes incompativeis.")

        result = []

        for i in range(rows_a):
            row = []
            for j in range(cols_b):
                value = sum(
                    self.data[i][k] * other.data[k][j]
                    for k in range(cols_a)
                )
                row.append(value)
            result.append(row)

        return Tensor(result)
