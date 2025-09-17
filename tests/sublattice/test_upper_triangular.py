import pytest
import numpy as np

from cnf.sublattice.upper_triangular import UpperTriangular
from cnf.sublattice import Fraction, Vector

def np_mats_eq(mat1, mat2):
    return np.all(mat1 == mat2)

@pytest.mark.parametrize(
    "k_vec,expected_mat",
    [
        # Row 1
        (Vector([Fraction(1,4), Fraction.zero(), Fraction.zero()]),
         np.array([
            [4, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (Vector([Fraction.zero(), Fraction(1,4), Fraction.zero()]),
         np.array([
            [1, 0, 0],
            [0, 4, 0],
            [0, 0, 1],
        ])),
        (Vector([Fraction(1,4), Fraction(1,4), Fraction.zero()]),
         np.array([
            [4, 3, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (Vector([Fraction(1,2), Fraction(1,4), Fraction.zero()]),
         np.array([
            [2, 1, 0],
            [0, 2, 0],
            [0, 0, 1],
        ])),
        (Vector([Fraction(3,4), Fraction(1,4), Fraction.zero()]),
         np.array([
            [4, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (Vector([Fraction(1,4), Fraction(1,2), Fraction.zero()]),
         np.array([
            [4, 2, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (Vector([Fraction(0,4), Fraction(0,4), Fraction(1,4)]),
         np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 4],
        ])),

        # # Row 2
        (Vector([Fraction(1,4), Fraction(0,4), Fraction(1,4)]),
         np.array([
            [4, 0, 3],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (Vector([Fraction(1,2), Fraction(0,4), Fraction(1,4)]),
         np.array([
            [2, 0, 1],
            [0, 1, 0],
            [0, 0, 2],
        ])),
        (Vector([Fraction(1,4), Fraction(1,4), Fraction.zero()]),
         np.array([
            [4, 3, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (Vector([Fraction(1,4), Fraction(1,4), Fraction.zero()]),
         np.array([
            [4, 3, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (Vector([Fraction(1,4), Fraction(1,4), Fraction(1,4)]),
         np.array([
            [4, 3, 3],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (Vector([Fraction(1,2), Fraction(1,4), Fraction(1,4)]),
         np.array([
            [2, 1, 0],
            [0, 2, 3],
            [0, 0, 1],
        ])),
        (Vector([Fraction(3,4), Fraction(1,4), Fraction(1,4)]),
         np.array([
            [4, 1, 1],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        # Row 3
        (Vector([Fraction(0,4), Fraction(1,2), Fraction(1,4)]),
         np.array([
            [1, 0, 0],
            [0, 2, 1],
            [0, 0, 2],
        ])),
        (Vector([Fraction(1,4), Fraction(1,2), Fraction(1,4)]),
         np.array([
            [4, 2, 3],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (Vector([Fraction(1,2), Fraction(1,2), Fraction(1,4)]),
         np.array([
            [2, 1, 1],
            [0, 1, 0],
            [0, 0, 2],
        ])),
        (Vector([Fraction(3,4), Fraction(1,2), Fraction(1,4)]),
         np.array([
            [4, 2, 1],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (Vector([Fraction(0,4), Fraction(3,4), Fraction(1,4)]),
         np.array([
            [1, 0, 0],
            [0, 4, 1],
            [0, 0, 1],
        ])),
        (Vector([Fraction(1,4), Fraction(3,4), Fraction(1,4)]),
         np.array([
            [4, 1, 3],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (Vector([Fraction(1,2), Fraction(3,4), Fraction(1,4)]),
         np.array([
            [2, 1, 0],
            [0, 2, 1],
            [0, 0, 1],
        ])),
        # Row 4
        (Vector([Fraction(3,4), Fraction(3,4), Fraction(1,4)]),
         np.array([
            [4, 3, 1],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (Vector([Fraction(1,4), Fraction(0,4), Fraction(1,2)]),
         np.array([
            [4, 0, 2],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (Vector([Fraction(0,4), Fraction(1,4), Fraction(1,2)]),
         np.array([
            [1, 0, 0],
            [0, 4, 2],
            [0, 0, 1],
        ])),
        (Vector([Fraction(1,4), Fraction(1,4), Fraction(1,2)]),
         np.array([
            [4, 3, 2],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (Vector([Fraction(1,2), Fraction(1,4), Fraction(1,2)]),
         np.array([
            [2, 1, 0],
            [0, 2, 2],
            [0, 0, 1],
        ])),
        (Vector([Fraction(3,4), Fraction(1,4), Fraction(1,2)]),
         np.array([
            [4, 1, 2],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (Vector([Fraction(1,4), Fraction(1,2), Fraction(1,2)]),
         np.array([
            [4, 2, 2],
            [0, 1, 0],
            [0, 0, 1],
        ])),
    ]
)
def test_can_instantiate_from_k_vec(k_vec, expected_mat):
    ut = UpperTriangular.from_k_vector(k_vec, 4)
    exact_eq = np_mats_eq(ut.matrix, expected_mat)
    if not exact_eq:
        inverse = np.linalg.inv(expected_mat).astype(np.int64)
        computed = ut.matrix.astype(np.int64)

        test = np.linalg.inv(expected_mat) @ ut.matrix
        # print(test)
        # for row in test:
        #     for entry in row:
        #         assert isinstance(entry, np.int64)
        
        assert np.abs(np.linalg.det(test)) == 1