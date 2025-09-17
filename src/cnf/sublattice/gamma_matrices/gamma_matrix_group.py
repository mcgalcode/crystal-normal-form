import numpy as np
from itertools import combinations_with_replacement, permutations

from .gamma_matrix_tuple import GammaMatrixTuple
from ...linalg.matrix_tuple import MatrixTuple
from ..primes import get_prime_parcels, get_prime_factors

from typing import Iterable

class GammaMatrixGroup():

    @classmethod
    def for_index(cls, N: int):
        prime_factors = get_prime_factors(N)
        parcels = get_prime_parcels(prime_factors, 3)
        matrix_group = cls()
        for parcel in parcels:
            
            first_row_possible_entries = range(parcel[0])
            possible_first_rows = [list(permutations(combo)) for combo in combinations_with_replacement(first_row_possible_entries, 2)]
            possible_first_rows = [perm for perm_list in possible_first_rows for perm in perm_list]
            possible_second_rows = range(parcel[1])
            for row1 in possible_first_rows:
                for row2 in possible_second_rows:
                    
                    gamma_matrix = np.diag(parcel)
                    print(row1, row2, gamma_matrix)
                    gamma_matrix[0][1] = row1[0]
                    gamma_matrix[0][2] = row1[1]
                    gamma_matrix[1][2] = row2
                    matrix_group.add_matrix(GammaMatrixTuple(gamma_matrix))
        
        return matrix_group

    
    def __init__(self, matrices: Iterable[GammaMatrixTuple] = None):
        if matrices is None:
            matrices = set()
        self.matrices = set(matrices)
    
    def add_matrix(self, mat):
        if isinstance(mat, np.ndarray):
            mat = GammaMatrixTuple(mat)
        
        if not self.contains_equivalent(mat):
            self.matrices.add(mat)
    
    def contains_exact(self, other):
        if isinstance(other, np.ndarray):
            other = MatrixTuple(other)
        return other in self.matrices
    
    def contains_equivalent(self, other):
        if isinstance(other, np.ndarray):
            other = GammaMatrixTuple(other)

        if self.contains_exact(other):
            return True

        for existing_mat in self.matrices:
            if existing_mat.generates_same_sublattice(other):
                return True
        
        return False
    
    def __len__(self):
        return len(self.matrices)
        