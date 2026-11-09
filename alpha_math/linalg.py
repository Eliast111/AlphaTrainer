"""Algebra linear com aceleracao NumPy e fallback Python puro."""
from alpha_math.backend import np

def _matrix(value, name):
    if np is not None:
        try:
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} deve ser uma matriz numerica.") from exc
        if arr.ndim != 2:
            raise ValueError(f"{name} deve ser bidimensional.")
        return arr
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} deve ser uma lista de listas.")
    rows = [list(row) for row in value]
    if rows and any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{name} tem linhas de tamanhos diferentes.")
    return rows

def transpose(matrix):
    if np is not None:
        return np.asarray(_matrix(matrix, "matrix").T).tolist()
    rows = _matrix(matrix, "matrix")
    return [list(column) for column in zip(*rows)] if rows else []

def matmul(a, b):
    """Multiplicacao matricial (n x k) @ (k x m), acelerada por NumPy."""
    left, right = _matrix(a, "a"), _matrix(b, "b")
    left_shape = left.shape if np is not None else (len(left), len(left[0]) if left else 0)
    right_shape = right.shape if np is not None else (len(right), len(right[0]) if right else 0)
    if left_shape[1] != right_shape[0]:
        raise ValueError("Dimensoes incompativeis para multiplicacao matricial.")
    if np is not None:
        return (left @ right).tolist()
    columns = list(zip(*right)) if right else []
    return [[sum(x * y for x, y in zip(row, col)) for col in columns] for row in left]

def add(a, b):
    left, right = _matrix(a, "a"), _matrix(b, "b")
    if np is not None:
        if left.shape != right.shape:
            raise ValueError("Dimensoes incompativeis para soma matricial.")
        return (left + right).tolist()
    if len(left) != len(right) or any(len(x) != len(y) for x, y in zip(left, right)):
        raise ValueError("Dimensoes incompativeis para soma matricial.")
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(left, right)]

def zeros_like(matrix):
    rows = _matrix(matrix, "matrix")
    if np is not None:
        return np.zeros_like(rows, dtype=float).tolist()
    return [[0.0] * len(row) for row in rows]

def column_sums(matrix):
    rows = _matrix(matrix, "matrix")
    if np is not None:
        return np.sum(rows, axis=0).astype(float).tolist()
    return [sum(column) for column in zip(*rows)] if rows else []
