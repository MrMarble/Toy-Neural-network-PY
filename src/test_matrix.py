import pytest
from matrix import Matrix
import json


@pytest.fixture()
def init_matrix():
    m = Matrix(3, 3)
    m.data[0] = [1, 2, 3]
    m.data[1] = [4, 5, 6]
    m.data[2] = [7, 8, 9]
    return m


@pytest.fixture()
def init_matrix_10():
    m = Matrix(3, 3)
    m.data[0] = [10, 11, 12]
    m.data[1] = [13, 14, 15]
    m.data[2] = [16, 17, 18]
    return m


def compare(matrix_a, matrix_b):
    assert matrix_a.rows == matrix_b.rows
    assert matrix_a.cols == matrix_b.cols
    assert (matrix_a.data == matrix_b.data).all()


def test_add_scalar_to_matrix(init_matrix):
    init_matrix.add(1)

    expected = Matrix(3, 3)
    expected.data[0] = [2, 3, 4]
    expected.data[1] = [5, 6, 7]
    expected.data[2] = [8, 9, 10]

    compare(init_matrix, expected)


def test_add_matrix_to_matrix(init_matrix, init_matrix_10):
    init_matrix.add(init_matrix_10)

    expected = Matrix(3, 3)
    expected.data[0] = [11, 13, 15]
    expected.data[1] = [17, 19, 21]
    expected.data[2] = [23, 25, 27]

    compare(init_matrix, expected)


def test_subtract_matrix_from_matrix(init_matrix, init_matrix_10):
    result = Matrix.subtract(init_matrix_10, init_matrix)

    expected = Matrix(3, 3)
    expected.data[0] = [9, 9, 9]
    expected.data[1] = [9, 9, 9]
    expected.data[2] = [9, 9, 9]

    compare(result, expected)


def test_from_list():
    result = Matrix.from_list([0, 0, 0])
    expected = Matrix(3, 1)

    compare(result, expected)


def test_transpose():
    m = Matrix(2, 2)
    m.data[0] = [1, 2]
    m.data[1] = [3, 4]
    m = Matrix.transpose(m)

    expected = Matrix(2, 2)
    expected.data[0] = [1, 3]
    expected.data[1] = [2, 4]
    compare(m, expected)


def test_matrix_product():
    m = Matrix(2, 3)
    m.data[0] = [1, 2, 3]
    m.data[1] = [4, 5, 6]
    n = Matrix(3, 2)
    n.data[0] = [7, 8]
    n.data[1] = [9, 10]
    n.data[2] = [11, 12]
    result = Matrix.multiply(m, n)

    expected = Matrix(2, 2)
    expected.data[0] = [58, 64]
    expected.data[1] = [139, 154]

    compare(result, expected)


def test_to_list(init_matrix):
    assert (init_matrix.to_list() == [1, 2, 3, 4, 5, 6, 7, 8, 9]).all()
