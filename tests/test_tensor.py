import unittest

from alpha_math.tensor import Tensor


class TestAlphaTensor(unittest.TestCase):

    def test_vector_addition(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        self.assertEqual(a.add(b).data, [5, 7, 9])

    def test_scalar_multiplication(self):
        a = Tensor([2, 3, 4])
        self.assertEqual(a.multiply(2).data, [4, 6, 8])

    def test_dot_product(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        self.assertEqual(a.dot(b), 32)

    def test_matrix_multiplication(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        self.assertEqual(
            a.matmul(b).data,
            [[19, 22], [43, 50]]
        )

    def test_matrix_shape(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(a.shape(), (2, 3))


if __name__ == "__main__":
    unittest.main()
