import pytest
import numpy as np

from cnf.lattice.vonorm_unimodular import reduce_col, is_voronoi_vector_column


def test_is_voronoi_vector_column():
    col = np.array([0, 1, 2])
    assert not is_voronoi_vector_column(col)

    col = np.array([0, 0, 1])
    assert is_voronoi_vector_column(col)

    col = np.array([0, 1, 0])
    assert is_voronoi_vector_column(col)

def test_reduce_col():

    col = np.array([1, 2, 1])
    print(reduce_col(col))