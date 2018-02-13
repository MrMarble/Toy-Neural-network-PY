import pytest
from matrix import Matrix


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

def test_add_scalar_to_matrix(init_matrix):
    init_matrix.add(1)

    expected = Matrix(3, 3)
    expected.data[0] = [2, 3, 4]
    expected.data[1] = [5, 6, 7]
    expected.data[2] = [8, 9, 10]

    assert init_matrix.rows == expected.rows
    assert init_matrix.cols == expected.cols
    assert (init_matrix.data == expected.data).all()


def test_add_matrix_to_matrix(init_matrix, init_matrix_10):
    init_matrix.add(init_matrix_10)

    expected = Matrix(3, 3)
    expected.data[0] = [11, 13, 15]
    expected.data[1] = [17, 19, 21]
    expected.data[2] = [23, 25, 27]

    assert init_matrix.rows == expected.rows
    assert init_matrix.cols == expected.cols
    assert (init_matrix.data == expected.data).all()


def test_subtract_matrix_from_matrix(init_matrix, init_matrix_10):
    result = Matrix.subtract(init_matrix_10, init_matrix)

    expected = Matrix(3, 3)
    expected.data[0] = [9, 9, 9]
    expected.data[1] = [9, 9, 9]
    expected.data[2] = [9, 9, 9]

    assert result.rows == expected.rows
    assert result.cols == expected.cols
    assert (result.data == expected.data).all()
