import pytest
import numpy as np

from cnf.sublattice.gamma_matrices import GammaMatrixGroup, GammaMatrixTuple

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

# def test_gamma_matrix_group(n_equals_4_generators):
#     matrices = GammaMatrixGroup.for_index(4)
#     for m in matrices:
#         print(m)
#     print(f"Found {len(matrices)} gamma matrices")