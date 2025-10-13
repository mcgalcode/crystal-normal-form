import pytest
import numpy as np

from cnf.sublattice.gamma_matrices import GammaMatrixTuple
from cnf.linalg.matrix_tuple import MatrixTuple

def np_mats_eq(mat1, mat2):
    return np.all(mat1 == mat2)

def test_equivalence():
    mat1 = GammaMatrixTuple(np.array([
        [2, 1, 0],
        [0, 2, 3],
        [0, 0, 1],
    ]))

    mat2 = GammaMatrixTuple(np.array([
        [2, 1, 1],
        [0, 2, 1],
        [0, 0, 1],
    ]))

    assert mat1.generates_same_sublattice(mat2)
    assert mat2.generates_same_sublattice(mat1)

    mat3 = GammaMatrixTuple(np.array([
        [2, 0, 1],
        [0, 2, 1],
        [0, 0, 1],
    ]))

    assert not mat1.generates_same_sublattice(mat3)
    assert not mat2.generates_same_sublattice(mat3)
    assert not mat3.generates_same_sublattice(mat1)
    assert not mat3.generates_same_sublattice(mat2)

    assert mat1.generates_same_sublattice(mat1)
    assert mat2.generates_same_sublattice(mat2)
    assert mat3.generates_same_sublattice(mat3)

def test_can_instantiate_from_k_vec(n_equals_4_generators_with_kvecs):
    tested = 0
    for pair in n_equals_4_generators_with_kvecs:
        k_vec, expected_mat = pair
        ut = GammaMatrixTuple.from_k_vector(k_vec, 4)
        exact_eq = np_mats_eq(ut.matrix, expected_mat)
        if not exact_eq:
            assert ut.generates_same_sublattice(GammaMatrixTuple(expected_mat))
        tested += 1
    
    assert tested == 28