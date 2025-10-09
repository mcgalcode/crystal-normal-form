from ..linalg import MatrixTuple

def is_unimodular_set_closed(unimodular_mats):
    mat_tuples = set([MatrixTuple(m).tuple for m in unimodular_mats])
    assert len(mat_tuples) == len(unimodular_mats)

    for m1 in unimodular_mats:
        for m2 in unimodular_mats:
            assert MatrixTuple(m1 @ m2).tuple in mat_tuples

def combine_unimodular_matrices(unimodular_mats):
    combined_mat = unimodular_mats[0]
    for mat in unimodular_mats[1:]:
        combined_mat = combined_mat @ mat
    return combined_mat