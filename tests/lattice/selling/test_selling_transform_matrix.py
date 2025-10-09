import numpy as np
from cnf.lattice.selling import SellingTransformMatrix

def test_transform_matrices():
    # I just computed these two with pen and paper
    expected_mat = np.array([
        [-1, 0, 1],
        [ 0, 1, 0],
        [ 0, 0, 1],
    ])
    constructed_mat = SellingTransformMatrix.from_pair((0,1))

    assert np.all(expected_mat == constructed_mat.matrix)

    expected_mat_2 = np.array([
        [ 1, 0,0],
        [ 1,-1,1],
        [ 0, 0,1],
    ])
    constructed_mat_2 = SellingTransformMatrix.from_pair((1,3))

    assert np.all(expected_mat_2 == constructed_mat_2.matrix)