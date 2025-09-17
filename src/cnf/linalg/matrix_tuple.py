import numpy as np

class MatrixTuple():

    @classmethod
    def from_tuple(cls, mat_tuple):
        return cls(np.array(mat_tuple).reshape((3,3)))

    def __init__(self, matrix: np.array):
        self.matrix = matrix
        entries = []
        for row in matrix:
            for val in row:
                entries.append(val)
        self.tuple = tuple([int(e) for e in entries])