import numpy as np
from enum import Enum

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

def apply_swaps(label_list: list[int], swap_series: list[tuple[int, int]]):
    for s in swap_series:
        apply_swap(label_list, s)

def apply_swap(label_list, swap: tuple[int, int]):
    idx_1, idx_2 = swap
    label_1 = label_list[idx_1]
    label_list[idx_1] = label_list[idx_2]
    label_list[idx_2] = label_1

def get_unimodular_matrix_from_voronoi_vectors(vectors: list[VoronoiVectors]):
    if not all([isinstance(v, VoronoiVectors) for v in vectors]):
        raise ValueError("Enum values VoronoiVectors required!")
    
    cols = [COLUMNS[v] for v in vectors]
    return np.array(cols).T

def get_unimodular_matrix_from_voronoi_vector_idxs(v_idxs: list[int]):
    vectors = [idx_to_voronoi_vec[i] for i in v_idxs]
    return get_unimodular_matrix_from_voronoi_vectors(vectors)

def get_unimodular_matrix_for_swap(swap: tuple[int, int]):
    # vectors v1,v2,and v3 are our lattice generators
    # v0 (the first one) forms the superbasis and v0=-v1-v2-v3
    vonorm_idx1, vonorm_idx2 = swap
    labels = [0,1,2,3,4,5,6]
    tmp = labels[vonorm_idx2]
    labels[vonorm_idx2] = labels[vonorm_idx1]
    labels[vonorm_idx1] = tmp

    # Now, in this new permutation, positions 1..3 hold
    # the labels that correspond to the new vectors
    return get_unimodular_matrix_from_voronoi_vector_idxs(labels[1:4])

def get_unimodular_matrix_for_swap_series(swap_series: list[tuple[int, int]]):
    labels = [0,1,2,3,4,5,6]
    apply_swaps(labels, swap_series)
    return get_unimodular_matrix_from_voronoi_vector_idxs(labels[1:4])

def is_unimodular_set_closed(unimodular_mats):
    mat_tuples = set([UnimodularMatrix(m).tuple for m in unimodular_mats])
    assert len(mat_tuples) == len(unimodular_mats)

    for m1 in unimodular_mats:
        for m2 in unimodular_mats:
            assert UnimodularMatrix(m1 @ m2).tuple in mat_tuples
class UnimodularMatrix():

    @classmethod
    def from_tuple(cls, mat_tuple):
        return cls(np.array(mat_tuple).reshape((3,3)))

    def __init__(self, matrix: np.array):
        self.matrix = matrix
        entries = []
        for row in matrix:
            for val in row:
                entries.append(val)
        self.tuple = tuple([int(e) for e in entries])