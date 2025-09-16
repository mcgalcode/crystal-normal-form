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

def test_get_unimodular_matrix_for_swap():
    # These tuples represent labels on voronoi vectors
    # to swap
    swap = (0, 1)
    mat = uni.get_unimodular_matrix_for_swap(swap)
    assert (mat.T[0] == [-1, -1, -1]).all()
    assert (mat.T[1] == [0, 1, 0]).all()
    assert (mat.T[2] == [0, 0, 1]).all()

    swap = (1, 3)
    mat = uni.get_unimodular_matrix_for_swap(swap)
    assert (mat.T[0] == [0, 0, 1]).all()
    assert (mat.T[1] == [0, 1, 0]).all()
    assert (mat.T[2] == [1, 0, 0]).all()

def test_apply_swap():
    label_list = [0, 1, 2, 3, 4, 5, 6]
    swap = (2, 3)
    uni.apply_swap(label_list, swap)
    assert label_list == [0, 1, 3, 2, 4, 5, 6]

    swap = (3, 6)
    uni.apply_swap(label_list, swap)
    assert label_list == [0, 1, 3, 6, 4, 5, 2]

    swap = (1, 2)
    uni.apply_swap(label_list, swap)
    assert label_list == [0, 3, 1, 6, 4, 5, 2]

    swap = (4, 5)
    uni.apply_swap(label_list, swap)
    assert label_list == [0, 3, 1, 6, 5, 4, 2]

def test_apply_swap_series():
    label_list = [0, 1, 2, 3, 4, 5, 6]
    swap_series = [(2, 3), (3, 6), (1, 2), (4, 5)]

    uni.apply_swaps(label_list, swap_series)
    assert label_list == [0, 3, 1, 6, 5, 4, 2]

    label_list = [0, 1, 2, 3, 4, 5, 6]
    swap_series = [(2, 4), (0, 6), (1, 6), (3, 5)]
    uni.apply_swaps(label_list, swap_series)
    assert label_list == [6, 0, 4, 5, 2, 3, 1]

def test_get_unimodular_matrix_for_swap_series():
    swap_series = [(2, 3), (3, 6), (1, 2), (4, 5)]
    mat = uni.get_unimodular_matrix_for_swap_series(swap_series)
    assert (mat.T[0] == [0, 0, 1]).all()
    assert (mat.T[1] == [1, 0, 0]).all()
    assert (mat.T[2] == [-1, -1, 0]).all()

    swap_series = [(2, 4), (0, 6), (1, 6), (3, 5)]
    # -> [6, 0, 4, 5, 2, 3, 1]
    mat = uni.get_unimodular_matrix_for_swap_series(swap_series)
    assert (mat.T[0] == [-1, -1, -1]).all()
    assert (mat.T[1] == [0, -1, -1]).all()
    assert (mat.T[2] == [-1, 0, -1]).all()
