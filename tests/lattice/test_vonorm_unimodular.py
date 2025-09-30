import pytest
import numpy as np

from cnf.lattice.vonorm_unimodular import reduce_col, is_voronoi_vector_column, get_unimodular_matrix_from_voronoi_vector_idxs
from cnf.lattice.permutations import VonormPermutation, VONORM_PERMUTATION_TO_CONORM_PERMUTATION, UnimodPermMapper

def test_get_unimodular_matrix_from_voronoi_vector_idxs():
    idxs = [2,1,3]
    mat = get_unimodular_matrix_from_voronoi_vector_idxs(idxs)
    assert (mat.T[0] == [0,1,0]).all()
    assert (mat.T[1] == [1,0,0]).all()
    assert (mat.T[2] == [0,0,1]).all()


    idxs = [0,3,2]
    mat = get_unimodular_matrix_from_voronoi_vector_idxs(idxs)
    assert (mat.T[0] == [-1,-1,-1]).all()
    assert (mat.T[1] == [0,0,1]).all()
    assert (mat.T[2] == [0,1,0]).all()

    idxs = [0,3,2]
    mat = get_unimodular_matrix_from_voronoi_vector_idxs(idxs)
    assert (mat.T[0] == [-1,-1,-1]).all()
    assert (mat.T[1] == [0,0,1]).all()
    assert (mat.T[2] == [0,1,0]).all()

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