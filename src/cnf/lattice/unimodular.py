import numpy as np
from enum import Enum
from ..linalg.matrix_tuple import MatrixTuple

class VoronoiVectors(Enum):

    V0 = 0
    V1 = 1
    V2 = 2
    V3 = 3
    V0_V1 = 4
    V0_V2 = 5
    V0_V3 = 6


COLUMNS = {
    VoronoiVectors.V1: [1, 0, 0],
    VoronoiVectors.V2: [0, 1, 0],
    VoronoiVectors.V3: [0, 0, 1],

    VoronoiVectors.V0: [-1, -1, -1],

    VoronoiVectors.V0_V1: [0, -1, -1],
    VoronoiVectors.V0_V2: [-1, 0, -1],
    VoronoiVectors.V0_V3: [-1, -1, 0]
}

idx_to_voronoi_vec = { v.value: v for v in VoronoiVectors}
voronoi_vec_to_idx = { v: v.value for v in VoronoiVectors}

def get_unimodular_matrix_from_voronoi_vectors(vectors: list[VoronoiVectors]):
    if not all([isinstance(v, VoronoiVectors) for v in vectors]):
        raise ValueError("Enum values VoronoiVectors required!")
    
    cols = [COLUMNS[v] for v in vectors]
    return np.array(cols).T

def get_unimodular_matrix_from_voronoi_vector_idxs(v_idxs: list[int]):
    vectors = [idx_to_voronoi_vec[i] for i in v_idxs]
    return get_unimodular_matrix_from_voronoi_vectors(vectors)

def is_unimodular_set_closed(unimodular_mats):
    mat_tuples = set([MatrixTuple(m).tuple for m in unimodular_mats])
    assert len(mat_tuples) == len(unimodular_mats)

    for m1 in unimodular_mats:
        for m2 in unimodular_mats:
            assert MatrixTuple(m1 @ m2).tuple in mat_tuples
