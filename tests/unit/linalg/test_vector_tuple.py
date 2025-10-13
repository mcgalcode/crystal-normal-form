import pytest
import numpy as np

from cnf.linalg import VectorTuple, MatrixTuple

def test_to_cols():
    mat = np.array([
        [1, 0, 0],
        [-1, -1, 1],
        [0, 0, -1]
    ])
    cols = MatrixTuple(mat).to_cols()
    assert cols[0] == VectorTuple([1, -1, 0])
    assert cols[1] == VectorTuple([0, -1, 0])
    assert cols[2] == VectorTuple([0, 1, -1])