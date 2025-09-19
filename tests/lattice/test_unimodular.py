import cnf.lattice.unimodular as uni
import numpy as np

def test_get_unimodular_matrix_from_voronoi_vector_idxs():
    idxs = [2,1,3]
    mat = uni.get_unimodular_matrix_from_voronoi_vector_idxs(idxs)
    assert (mat.T[0] == [0,1,0]).all()
    assert (mat.T[1] == [1,0,0]).all()
    assert (mat.T[2] == [0,0,1]).all()


    idxs = [0,3,2]
    mat = uni.get_unimodular_matrix_from_voronoi_vector_idxs(idxs)
    assert (mat.T[0] == [-1,-1,-1]).all()
    assert (mat.T[1] == [0,0,1]).all()
    assert (mat.T[2] == [0,1,0]).all()

    idxs = [0,3,2]
    mat = uni.get_unimodular_matrix_from_voronoi_vector_idxs(idxs)
    assert (mat.T[0] == [-1,-1,-1]).all()
    assert (mat.T[1] == [0,0,1]).all()
    assert (mat.T[2] == [0,1,0]).all()