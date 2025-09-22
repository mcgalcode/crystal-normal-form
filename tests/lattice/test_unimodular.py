import pytest

import cnf.lattice.vonorm_unimodular as uni
import numpy as np

from cnf.lattice.permutations import VonormPermutation, VONORM_PERMUTATION_TO_CONORM_PERMUTATION

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

@pytest.mark.xfail
def test_two_permutation_umats_create_another_umat():
    all_perms = list(VONORM_PERMUTATION_TO_CONORM_PERMUTATION.keys())
    # all_perms = [p for p in all_perms if VonormPermutation(p).to_conorm_permutation().perm[-1] == 6]
    print(len(all_perms))
    for p1 in all_perms:
        p1 = VonormPermutation(p1)
        for p2 in all_perms:
            p2 = VonormPermutation(p2)

            # print(f"P1: {p1}")
            # print(f"P2: {p2}")
            # print(f"P1 * P2: {p1.compose(p2)}")

            p1_umat = p1.to_unimodular_matrix()
            p2_umat = p2.to_unimodular_matrix()
            composed_umat = p1.compose(p2).to_unimodular_matrix()
            # print(f"P1 Umat: \n{p1_umat}")
            # print(f"P2 Umat: \n{p2_umat}")
            # print(f"Composed Perm Umat: \n{composed_umat}")
            # print(f"Multipled Umat: \n{p2_umat @ p1_umat}")

            assert np.all(composed_umat == p2_umat @ p1_umat)