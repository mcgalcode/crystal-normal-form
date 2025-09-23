import numpy as np

from ...linalg.matrix_tuple import MatrixTuple
from ..voronoi.vector_pair import VoronoiVectorPair

LABELS_TO_COLS = {
    1: np.array([1, 0, 0]),
    2: np.array([0, 1, 0]),
    3: np.array([0, 0, 1]),
    0: np.array([-1, -1, -1]),
}

class SellingTransformMatrix(MatrixTuple):

    @staticmethod
    def from_pair(pair: VoronoiVectorPair):
        return SELLING_TRANSFORM_MATRICES[pair]
    
    @staticmethod
    def inverse_from_pair(pair: VoronoiVectorPair):
        return SELLING_TRANSFORM_INVERSE_MATRICES[pair]

def build_st_mat_from_pair(pair: VoronoiVectorPair):
    columns = []
    i, j = pair
    # We only care about vectors 1,2, and 3
    for lattice_vec_label in range(1,4):
        if lattice_vec_label == i:
            col = - LABELS_TO_COLS[lattice_vec_label]
        elif lattice_vec_label == j:
            col = LABELS_TO_COLS[lattice_vec_label]
        else:
            col = LABELS_TO_COLS[lattice_vec_label] + LABELS_TO_COLS[i]
        columns.append(col)
    return SellingTransformMatrix(np.array(columns).T)

SELLING_TRANSFORM_MATRICES: dict[VoronoiVectorPair, SellingTransformMatrix] = {}
SELLING_TRANSFORM_INVERSE_MATRICES: dict[VoronoiVectorPair, SellingTransformMatrix] = {}

for p in VoronoiVectorPair.CANONICAL_PAIRS:
    SELLING_TRANSFORM_MATRICES[p] = build_st_mat_from_pair(p)

for p in VoronoiVectorPair.CANONICAL_PAIRS:
    SELLING_TRANSFORM_INVERSE_MATRICES[p] = SellingTransformMatrix(SELLING_TRANSFORM_MATRICES[p].inverse())