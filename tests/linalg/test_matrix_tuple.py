import pytest
import numpy as np

from cnf.lattice.unimodular import get_unimodular_matrix_for_swap
from cnf.linalg.matrix_tuple import MatrixTuple

def test_roundtrip_unimodular_class():
    example_unimodular = get_unimodular_matrix_for_swap((0,1))
    before = MatrixTuple(example_unimodular)
    after = MatrixTuple.from_tuple(before.tuple)

    assert np.all(before.matrix == after.matrix)