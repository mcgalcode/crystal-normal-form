import numpy as np

from ...linalg.matrix_tuple import MatrixTuple
from ..fraction_vector import FractionVector
from ..fraction import Fraction

class GammaMatrixTuple(MatrixTuple):

    @classmethod
    def from_k_vector(cls, k_vec: FractionVector, N):

        mat = np.zeros((3,3))

        # First column is given by the denominator on the first
        # coordinate (or 1 if the first entry is zero)
        if k_vec.coords[0] == Fraction.zero():
            mat[0][0] = 1
        else:
            mat[0][0] = k_vec.coords[0].denom

        # Second column i = 2, and j = {1}
        # The goal here is to find a linear combination (z1 * c1) + (z2 * c2)
        # that a) keeps z2 as low as possible
        # and  b) yields an integer
        # Start by iterating through z2 values:
        c1 = k_vec.coords[0]
        c2 = k_vec.coords[1]
        c3 = k_vec.coords[2]

        if c2 == Fraction.zero():
            mat[1][1] = 1
        else:
            col_2_found = False
            for z2 in range(1, N + 1):
                for z1 in range(0, N):
                    if c1.is_zero():
                        z1 = 0
                    combination = c1.multiply(z1).add(c2.multiply(z2))
                    # print(f"{z1} * {c1} + {z2} * {c2} = {combination}, is_int={combination.is_int()}")
                    if combination.is_int():
                        mat[0][1] = z1
                        mat[1][1] = z2
                        col_2_found = True
                        break
                if col_2_found:
                    break
            if not col_2_found:
                raise RuntimeError("Couldn't find valid combination for column 2")
            
        if c3 == Fraction.zero():
            mat[2][2] = 1           
        else:
            col_3_found = False
            for z3 in range(1, N + 1):
                for z2 in range(0, N):
                    for z1 in range(0, N):
                        if c1.is_zero():
                            z1 = 0
                        
                        if c2.is_zero():
                            z2 = 0
                        combination = c1.multiply(z1).add(c2.multiply(z2)).add(c3.multiply(z3))
                        # print(f"{z1} * {c1} + {z2} * {c2} + {z3} * {c3} = {combination}, is_int={combination.is_int()}")
                        if combination.is_int():
                            mat[0][2] = z1
                            mat[1][2] = z2
                            mat[2][2] = z3
                            col_3_found = True
                            break
                    if col_3_found:
                        break
                if col_3_found:
                        break
        
        return cls(mat)

    def generates_same_sublattice(self, other: 'GammaMatrixTuple'):
        return MatrixTuple(self.inverse() @ other.matrix).is_unimodular()