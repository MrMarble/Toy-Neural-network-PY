import pytest
from matrix import Matrix


def test_add_scalar_to_matrix(): 
    m = Matrix(3, 3)
    m.data[0] = [1, 2, 3]
    m.data[1] = [4, 5, 6]
    m.data[2] = [7, 8, 9]
    m.add(1)

    expected = Matrix(3, 3)
    expected.data[0] = [2, 3, 4]
    expected.data[1] = [5, 6, 7]
    expected.data[2] = [8, 9, 10]

    assert m.rows == expected.rows
    assert m.cols == expected.cols
    assert (m.data == expected.data).all()


def test_add_matrix_to_matrix():
    m = Matrix(2, 2)
    m.data[0] = [1, 2]
    m.data[1] = [3, 4]
    n = Matrix(2, 2)
    n.data[0] = [10, 11]
    n.data[1] = [12, 13]
    m.add(n)

    expected = Matrix(2, 2)
    expected.data[0] = [11, 13]
    expected.data[1] = [15, 17]

    assert m.rows == expected.rows
    assert m.cols == expected.cols
    assert (m.data == expected.data).all()


def test_subtract_matrix_from_matrix():
    m = Matrix(2, 2)
    m.data[0] = [10, 11]
    m.data[1] = [12, 13]
    n = Matrix(2, 2)
    n.data[0] = [1, 2]
    n.data[1] = [3, 4]
    m_minus_n = Matrix.substract(m, n)

    expected = Matrix(2, 2)
    expected.data[0] = [9, 9]
    expected.data[1] = [9, 9]

    assert m_minus_n.rows == expected.rows
    assert m_minus_n.cols == expected.cols
    assert (m_minus_n.data == expected.data).all()
