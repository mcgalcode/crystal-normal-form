import pytest
import numpy as np

from cnf.lattice.swaps.utils import get_unimodular_matrix_for_swap
from cnf.linalg.matrix_tuple import MatrixTuple

def test_roundtrip_unimodular_class():
    example_unimodular = get_unimodular_matrix_for_swap((0,1))
    before = MatrixTuple(example_unimodular)
    after = MatrixTuple.from_tuple(before.tuple)

    assert np.all(before.matrix == after.matrix)

def test_equality_function():
    mat = np.array([
        [4, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    m1 = MatrixTuple(mat)
    m2 = MatrixTuple(mat)
    assert m1.tuple == m2.tuple

    container = set()
    container.add(m1)
    assert m2 in container

def test_unimodularity_check():
    example_unimodular = get_unimodular_matrix_for_swap((0,1))
    mt = MatrixTuple(example_unimodular)
    assert mt.is_unimodular()

    not_unimodular = get_unimodular_matrix_for_swap((0,1))
    not_unimodular[0][0] = 120
    mt = MatrixTuple(not_unimodular)
    assert not mt.is_unimodular()

    not_unimodular = np.random.random((3,3))
    mt = MatrixTuple(not_unimodular)
    assert not mt.is_unimodular()

def test_matrix_multiplication():
    npmat1 = np.array([
        [4, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    npmat2 = np.array([
        [2, 5, 0],
        [1, 3, 9],
        [6, 1, 1],
    ])

    mattuple1 = MatrixTuple(npmat1)
    mattuple2 = MatrixTuple(npmat2)

    assert np.all((mattuple1 @ mattuple2).matrix == npmat1 @ npmat2)

def test_sign_flip():
    npmat1 = np.array([
        [4, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    mattup1 = MatrixTuple(npmat1)
    flipped = mattup1.flip_signs()
    assert (flipped.matrix == -npmat1).all()
