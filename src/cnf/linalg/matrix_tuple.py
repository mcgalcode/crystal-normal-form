import numpy as np
from .utils import is_unimodular
from .vector_tuple import VectorTuple
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
    
    def determinant(self):
        return np.linalg.det(self.matrix)
    
    def flip_signs(self):
        return self.__class__(-self.matrix)
    
    def to_list(self):
        return list(self.tuple)
    
    def is_identity(self):
        return np.all(self.matrix == np.eye(3))
    
    def to_cols(self) -> list[VectorTuple]:
        cols = []
        for col in self.matrix.T:
            cols.append(VectorTuple(col))
        return cols
    
    def __hash__(self):
        return self.tuple.__hash__()
    
    def __eq__(self, other: 'MatrixTuple'):
        return self.tuple == other.tuple
    
    def __matmul__(self, other: 'MatrixTuple'):
        return self.__class__(self.matrix @ other.matrix)
    
    def __repr__(self):
        return self.tuple.__repr__()