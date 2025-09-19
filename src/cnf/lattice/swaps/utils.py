from ..unimodular import get_unimodular_matrix_from_voronoi_vector_idxs

def get_unimodular_matrix_for_swap_series(swap_series: list[tuple[int, int]]):
    labels = [0,1,2,3,4,5,6]
    apply_swaps(labels, swap_series)
    return get_unimodular_matrix_from_voronoi_vector_idxs(labels[1:4])

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

def apply_swaps(label_list: list[int], swap_series: list[tuple[int, int]]):
    for s in swap_series:
        apply_swap(label_list, s)

def apply_swap(label_list, swap: tuple[int, int]):
    idx_1, idx_2 = swap
    label_1 = label_list[idx_1]
    label_list[idx_1] = label_list[idx_2]
    label_list[idx_2] = label_1
