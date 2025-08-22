from itertools import permutations
import numpy as np

NUM_SUPERBASIS_VECS = 4
_UNIMODULAR_MATRIX_LOOKUP_BY_TUPLE = {}
_PERMUTATION_TUPLE_TO_INT = {}
_UNIMODULAR_MATRIX_LOOKUP_BY_INT = {}

def build_unimodular_matrices():

    vector_idxs = range(NUM_SUPERBASIS_VECS)
    idx_permutations = permutations(vector_idxs, NUM_SUPERBASIS_VECS)


    for permutation_idx, permutation in enumerate(idx_permutations):
        # We are concerned with the trailing three vectors in the permutation:
        # {p_0, p_1, p_2, p_3}
        # So each row of this matrix will correspond to the "recipe" to make
        # the vector corresponding to the idx p_x where x=1, 2, or 3.
        unimodular_matrix = np.zeros((3,3), dtype=int)
        for col_idx in range(3):
            # use col_idx + 1 because we are looking at the values
            # at positions 1..3 in the permutation to get the
            # vectors 1..3 (leaving off 0)
            new_vector_idx = permutation[col_idx + 1]
            if new_vector_idx == 0:
                # If the permutation requires that this row take
                # on the value of the 0th vector, we know that that
                # is the negative sum of all three "real basis vectors" (i.e. vectors 1-3)
                unimodular_matrix[:, col_idx] = -1
            else:
                # Take 1 of whatever vector is pointed to by the permutation
                # e.g., if the permutation is [2, 3, 0, 1]
                # the 3rd row should be [1, 0, 0], and the first row
                # should be [0, 0, 1]
                unimodular_matrix[new_vector_idx - 1, col_idx] = 1

        _PERMUTATION_TUPLE_TO_INT[permutation] = permutation_idx
        _UNIMODULAR_MATRIX_LOOKUP_BY_TUPLE[tuple(permutation)] = unimodular_matrix
        _UNIMODULAR_MATRIX_LOOKUP_BY_INT[permutation_idx] = unimodular_matrix

def get_unimodular_matrix_by_permutation_tuple(permutation_tuple: tuple):
    return _UNIMODULAR_MATRIX_LOOKUP_BY_TUPLE[permutation_tuple]

def get_unimodular_matrix_by_int(num: int):
    return _UNIMODULAR_MATRIX_LOOKUP_BY_INT[num]

def get_int_from_permutation_tuple(permutation_tuple: tuple):
    return _PERMUTATION_TUPLE_TO_INT[permutation_tuple]

