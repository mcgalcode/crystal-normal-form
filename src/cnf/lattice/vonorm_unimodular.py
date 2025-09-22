import numpy as np

from ..linalg import MatrixTuple, VectorTuple

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .permutations import VonormPermutation

VORONOI_IDX_TO_COLUMN = {
    0: VectorTuple([-1, -1, -1]),

    1: VectorTuple([1, 0, 0]),
    2: VectorTuple([0, 1, 0]),
    3: VectorTuple([0, 0, 1]),

    4: VectorTuple([0, -1, -1]),
    5: VectorTuple([-1, 0, -1]),
    6: VectorTuple([-1, -1, 0])
}

VORONOI_IDX_TO_COLUMNS_TO_SUM = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [0,1],
    5: [0,2],
    6: [0,3],
}

COLUMN_TO_VORONOI_IDX = { v: k for k, v in VORONOI_IDX_TO_COLUMN.items()}
ALL_VORONOI_VECTOR_COLS = set(COLUMN_TO_VORONOI_IDX.keys())

def get_unimodular_matrix_from_voronoi_vector_idxs(v_idxs: list[int]):
    for idx in v_idxs:
        if idx not in VORONOI_IDX_TO_COLUMN:
            raise ValueError(f"Cannot convert invalid Voronoi column idx {idx} to unimodular column!")    
    cols = [VORONOI_IDX_TO_COLUMN[i].vector for i in v_idxs]
    return np.array(cols).T

def reduce_col(col: np.ndarray):
    reduced_col = np.zeros(col.shape)
    for idx, value in enumerate(col):
        voronoi_idx = idx + 1
        voronoi_vec = VORONOI_IDX_TO_COLUMN[voronoi_idx]
        col = voronoi_vec.vector * value
        print(f"Adding {col}")
        reduced_col = reduced_col + col
    return reduced_col

def is_voronoi_vector_column(col: np.ndarray):
    return VectorTuple(col) in ALL_VORONOI_VECTOR_COLS

class VonormPermutationMatrix(MatrixTuple):

    @classmethod
    def from_permutation(cls, perm: 'VonormPermutation'):
        return cls.from_vector_idxs(perm.perm[1:4])

    @classmethod
    def from_vector_idxs(cls, v_idxs: list[int]):
        mat = get_unimodular_matrix_from_voronoi_vector_idxs(v_idxs)
        return cls(mat)

                                
