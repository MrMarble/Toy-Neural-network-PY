import numpy as np


class Matrix():

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.data = np.zeros((rows, cols))

    @staticmethod
    def from_list(arr: list):
        return Matrix(len(arr), 1).map(lambda _, i, j: arr[i])

    @staticmethod
    def subtract(matrix_a, matrix_b):
        if matrix_a.rows is not matrix_b.rows or matrix_a.cols is not matrix_b.cols:
            print('Columns and Rows of A must match Columns and Rows of B.')
            return
        return Matrix(matrix_a.rows, matrix_a.cols).map(lambda _, i, j: matrix_a.data[i][j] - matrix_b.data[i][j])

    @staticmethod
    def transpose(matrix):
        return Matrix(matrix.cols, matrix.rows).map(lambda _, i, j: matrix.data[j][i])

    def add(self, n: int):
        if type(n) is Matrix:
            if self.rows is not n.rows or self.cols is not n.cols:
                print('Columns and Rows of A must match Columns and Rows of B.')
                return
            return self.map(lambda e, i, j: e+n.data[i][j])
        else:
            return self.map(lambda e, i, j: e + n)

    def map(self, func):
        for row in range(self.rows):
            for col in range(self.cols):
                tmp = self.data[row][col]
                self.data[row][col] = func(tmp, row, col)
        return self
