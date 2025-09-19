import numpy as np
from .utils import is_unimodular
class MatrixTuple():

    @classmethod
    def from_tuple(cls, mat_tuple):
        return cls(np.array(mat_tuple).reshape((3,3)))

    def __init__(self, matrix: np.array):
        self.matrix = np.copy(matrix)
        entries = []
        for row in matrix:
            for val in row:
                entries.append(val)
        self.tuple = tuple([int(e) for e in entries])
    
    def is_unimodular(self):
        return is_unimodular(self.matrix)
    
    def inverse(self):
        return np.linalg.inv(self.matrix)
    
    def __hash__(self):
        return self.tuple.__hash__()
    
    def __eq__(self, other: 'MatrixTuple'):
        return self.tuple == other.tuple
    
    def __matmul__(self, other: 'MatrixTuple'):
        return MatrixTuple(self.matrix @ other.matrix)
    
    def __repr__(self):
        return self.tuple.__repr__()