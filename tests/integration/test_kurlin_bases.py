import pytest
import numpy as np

from itertools import permutations, product

from cnf.linalg import MatrixTuple
from cnf.lattice.permutations import UnimodPermMapper
from cnf.lattice.voronoi import ConormListForm
from cnf.lattice.voronoi.constants import CONORM_IDX_TO_PAIR

def get_zero_sets_for_voronoi_class(v_class):
    def _mask_selector(zero_mask):
        return ConormListForm(zero_mask).voronoi_class == v_class
    
    relevant_zero_sets = [zs for zs in UnimodPermMapper.all_zero_sets() if _mask_selector(zs)]
    return relevant_zero_sets

def find_matching_perms(mat_tup, voronoi_search_class):
    matching_permutations = []
    relevant_zero_sets = get_zero_sets_for_voronoi_class(voronoi_search_class)

    for zs in relevant_zero_sets:
        clf = ConormListForm(zs)
        perm_matrix_map = UnimodPermMapper.get_perms_for_zero_set(zs)
        for perm in perm_matrix_map:
            canonical_matrix = clf.canonical_matrix_for_perm(perm)
            if canonical_matrix == mat_tup:
                matching_permutations.append(perm)  
    matching_permutations = set(matching_permutations)

    return matching_permutations

def _assert_kurlin_matrix_present(kurlin_mat, voronoi_search_class):
    kurlin_tup = MatrixTuple(kurlin_mat)
    if kurlin_tup.determinant() == -1:
        kurlin_tup = MatrixTuple(-kurlin_mat)

    assert kurlin_tup.is_unimodular()
    assert kurlin_tup.determinant() == 1
    
    matching_permutations = find_matching_perms(kurlin_tup, voronoi_search_class)
    
    print(f"Found {len(matching_permutations)} permutation corresponding to Kurlin's example basis for Voronoi class V{voronoi_search_class}")
    assert len(matching_permutations) >= 1



def test_kurlin_v2_basis_represented_in_matrices():
    # pp. 9 of "A complete isometry classification..."

    kurlin_type_v2_mat = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, -1]
    ])

    _assert_kurlin_matrix_present(kurlin_type_v2_mat, 2)

def test_kurlin_v3_basis_represented_in_matrices():
    # pp. 11 of "A complete isometry classification..."
    kurlin_type_v3_mat_1 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, -1]
    ])

    _assert_kurlin_matrix_present(kurlin_type_v3_mat_1, 3)

    kurlin_type_v3_mat_2 = np.array([
        [-1, 1, 1],
        [0, 1, 0],
        [0, 0, 1]
    ])

    _assert_kurlin_matrix_present(kurlin_type_v3_mat_2, 3)


def test_kurlin_v4_basis_represented_in_matrices():
    # pp. 11 of "A complete isometry classification..."
    kurlin_type_v4_mat_1 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, -1]
    ])

    _assert_kurlin_matrix_present(kurlin_type_v4_mat_1, 4)

    kurlin_type_v4_mat_2 = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [1, 0, -1]
    ])

    _assert_kurlin_matrix_present(kurlin_type_v4_mat_2, 4)

def test_kurlin_v5_basis_represented_in_matrices():
    # pp. 11 of "A complete isometry classification..."

    def get_kurlin_matrices_from_zero_set(zeros):
        vector_idxs = [vector_idx for i in zeros for vector_idx in CONORM_IDX_TO_PAIR[i]]
        pairwise_orthogonal_vecs = set(vector_idxs)
        if len(pairwise_orthogonal_vecs) != 3:
            return []
        # assert len(pairwise_orthogonal_vecs) == 3

        coeffs = [-1, 1]
        coeff_sets = list(product([-1, 1], repeat=3))
        matrices = []
        for label_set in permutations(pairwise_orthogonal_vecs):
            for coeffs in coeff_sets:
                i, j, k = label_set
                i_coeff, j_coeff, k_coeff = coeffs

                col1 = np.zeros(3)
                col1[j - 1] = j_coeff

                col2 = np.zeros(3)
                col2[k - 1] = k_coeff
                col2[i - 1] = -i_coeff

                col3 = np.zeros(3)
                col3[k - 1] = -k_coeff
                col3[j - 1] = -j_coeff
                mat = np.array([col1, col2, col3]).T

                if np.linalg.det(mat) == 1 or np.linalg.det(-mat) == 1:
                    matrices.append((label_set, coeffs, mat))
        
        return matrices


    zero_sets = get_zero_sets_for_voronoi_class(5)
    positive_det_exists = []
    k1_assertions = 0
    k2_assertions = 0
    k3_assertions = 0
    for zs in zero_sets:
        mats = get_kurlin_matrices_from_zero_set(zs)
        # From Lemma 4.5 in Kurlin, we want to assert that 
        # at least one of these matrices has a matching perm
        # for k = 1, 2, 3
        
        if len(mats) > 0:
            positive_det_exists.append(zs)            
            k1_found = False
            k2_found = False
            k3_found = False

            k1_present = False
            k2_present = False
            k3_present = False
            for m in mats:
                ijk_labels, coeffs, matrix = m
                k = ijk_labels[2]
                if k == 1:
                    k1_present = True
                if k == 2:
                    k2_present = True
                if k == 3:
                    k3_present = True

                matrix = MatrixTuple(matrix)
                matching_perms = find_matching_perms(matrix, 5)
                if len(matching_perms) > 0:
                    if k == 1:
                        k1_found = True
                    if k == 2:
                        k2_found = True
                    if k == 3:
                        k3_found = True



            if k1_present:
                assert k1_found
                k1_assertions += 1
            
            if k2_present:
                assert k2_found
                k2_assertions += 1
            
            if k3_present:
                assert k3_found
                k3_assertions += 1

    print(f"Made {k1_assertions + k2_assertions + k3_assertions} assertions about procedurally generated Kurlin V5 matrices.")
    print(f"Made {k1_assertions} assertions about k = 1 cases.")
    print(f"Made {k2_assertions} assertions about k = 2 cases.")
    print(f"Made {k3_assertions} assertions about k = 3 cases.")
                
    # These are from v3
    kurlin_type_v5_mat_1 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, -1]
    ])

    _assert_kurlin_matrix_present(kurlin_type_v5_mat_1, 5)

    kurlin_type_v5_mat_2 = np.array([
        [-1, 1, 1],
        [0, 1, 0],
        [0, 0, 1]
    ])

    _assert_kurlin_matrix_present(kurlin_type_v5_mat_2, 5)

    # These ones are from V4
    kurlin_type_v5_mat_3 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, -1]
    ])

    _assert_kurlin_matrix_present(kurlin_type_v5_mat_3, 5)

    kurlin_type_v5_mat_4 = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [1, 0, -1]
    ])

    _assert_kurlin_matrix_present(kurlin_type_v5_mat_4, 5)

    # v_i = v_1
    # v_j = v_2
    # v_k = v_


    kurlin_type_v5_mat_5 = np.array([
        [0, -1, 0],
        [1, 0, -1],
        [0, 1, -1]
    ])

    _assert_kurlin_matrix_present(kurlin_type_v5_mat_5, 5)
