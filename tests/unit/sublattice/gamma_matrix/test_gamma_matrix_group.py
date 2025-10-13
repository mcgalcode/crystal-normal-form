import pytest
import numpy as np

from cnf.sublattice.gamma_matrices import GammaMatrixGroup, GammaMatrixTuple
from cnf.sublattice.kvec.kvec_generating_set import KVecGeneratingSet

def test_group_can_add_matrix():
    group = GammaMatrixGroup()
    mat = np.array([
        [4, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    assert not group.contains_equivalent(mat)
    assert not group.contains_exact(mat)
    group.add_matrix(mat)
    assert len(group) == 1
    assert group.contains_equivalent(mat)
    assert group.contains_exact(mat)

def test_group_adds_identical_matrix_only_once():
    group = GammaMatrixGroup()
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

    group.add_matrix(mat1)
    group.add_matrix(mat2)
    group.add_matrix(mat3)
    assert len(group) == 2

    assert group.contains_equivalent(mat1)
    assert group.contains_exact(mat1)
    assert group.contains_equivalent(mat2)
    assert not group.contains_exact(mat2)
    assert group.contains_equivalent(mat3)
    assert group.contains_exact(mat3)


def test_group_adds_equivalent_matrix_only_once():
    group = GammaMatrixGroup()
    mat = np.array([
        [4, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    mat2 = np.array([
        [4, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    mat3 = np.array([
        [4, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    group.add_matrix(mat)
    group.add_matrix(mat2)
    group.add_matrix(mat3)
    assert len(group) == 1
    assert group.contains_equivalent(mat)
    assert group.contains_exact(mat)

def test_gamma_matrices_are_invertible():
    for N in range(1,10):
        group = GammaMatrixGroup.for_index(N)
        for mat in group.matrices:
            assert mat.inverse() is not None

def test_gamma_matrix_group(all_n_equals_4_generators, n_equals_4_generators_from_kvecs, n_equals_4_generators_not_from_kvecs):
    N = 4
    # Build generators from kvecs
    kvec_group = GammaMatrixGroup()
    kvec_generating_set = KVecGeneratingSet.from_sublattice_index(N)
    for kvec in kvec_generating_set.representatives:
        gmat = GammaMatrixTuple.from_k_vector(kvec, N)
        kvec_group.add_matrix(gmat)
    
    assert len(kvec_group) == len(n_equals_4_generators_from_kvecs)
    # Build generators using upper triangular algorithm
    ut_group = GammaMatrixGroup.for_index(N)
    assert len(ut_group) == len(all_n_equals_4_generators)

    assert len(all_n_equals_4_generators) == 35

    for generator in n_equals_4_generators_from_kvecs:
        generator = GammaMatrixTuple(generator)
        assert kvec_group.contains_equivalent(generator)
        assert ut_group.contains_equivalent(generator)
    
    for generator in n_equals_4_generators_not_from_kvecs:
        generator = GammaMatrixTuple(generator)
        assert not kvec_group.contains_equivalent(generator)
        assert ut_group.contains_equivalent(generator)