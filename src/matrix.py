import numpy as np


class Matrix():

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.data = np.zeros((rows, cols))

    def add(self, n: int):
        if type(n) is Matrix:
            if self.rows is not n.rows or self.cols is not n.cols:
                print('Columns and Rows of A must match Columns and Rows of B.')
                return
            return self.map(lambda e, i, j: e+n.data[i][j])
        else:
            return self.map(lambda e, i, j: e + n)

    @staticmethod
    def subtract(a, b):
        if a.rows is not b.rows or a.cols is not b.cols:
            print('Columns and Rows of A must match Columns and Rows of B.')
            return
        return Matrix(a.rows, a.cols).map(lambda _, i, j: a.data[i][j] - b.data[i][j])

    def map(self, func):
        for row in range(self.rows):
            for col in range(self.cols):
                tmp = self.data[row][col]
                self.data[row][col] = func(tmp, row, col)
        return self
