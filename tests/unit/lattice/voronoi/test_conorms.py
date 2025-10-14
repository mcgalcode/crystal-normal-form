import numpy as np
import itertools

from cnf.linalg import MatrixTuple
from cnf.lattice import Superbasis
from cnf.lattice.permutations import ConormPermutation, VonormPermutation
from cnf.lattice.voronoi import ConormListForm, ConormList
from cnf.linalg.unimodular import UNIMODULAR_MATRICES

cols = [
    [-1, -1, -1],
    [1, 0 , 0],
    [0, 1, 0],
    [0, 0, 1]
]

def make_all_s4_matrices():
    mats = []
    for cs in itertools.permutations(cols, 3):
        mats.append(MatrixTuple.from_tuple(tuple(cs[0] + cs[1] + cs[2])))
    return mats

def superbases_equal(sb1, sb2, atol=1e-10, decimals=10):
    """
    Check if two superbases contain the same vectors using set comparison.
    
    Parameters
    ----------
    sb1, sb2 : array-like, shape (4, 3)
        Two superbases, each containing 4 vectors in 3D space
    atol : float
        Absolute tolerance for floating point comparison
    decimals : int
        Number of decimal places to round to for hashing
        
    Returns
    -------
    bool
        True if superbases contain the same vectors (in any order)
    """
    sb1 = np.asarray(sb1)
    sb2 = np.asarray(sb2)
    
    if sb1.shape != (4, 3) or sb2.shape != (4, 3):
        raise ValueError("Superbases must have shape (4, 3)")
    
    # Round vectors to create hashable tuples
    # This allows us to use set comparison
    def vectorize(sb):
        return set(tuple(np.round(v, decimals=decimals)) for v in sb)
    
    set1 = vectorize(sb1)
    set2 = vectorize(sb2)
    
    return set1 == set2

def _assert_all_representative_permutation_matrices_maintain_superbasis(superbasis: Superbasis):
    original_vonorms = superbasis.compute_vonorms()
    original_conorms = original_vonorms.conorms
    tested = []
    distinct_superbases = {}

    for cperm_with_mats in original_conorms.form.permissible_permutations():
        # print(mat, cperm.to_vonorm_permutation())
        cperm = cperm_with_mats.perm
        assert np.linalg.det(cperm_with_mats.matrix.matrix) == 1.0
        permuted = superbasis.apply_matrix_transform(cperm_with_mats.matrix.matrix)
        # print(f"Testing mat: {permuted.v0()}")
        found = False
        for sb_class_id, data in distinct_superbases.items():
            known_sb = data["known_sb"]
            if superbases_equal(known_sb.superbasis_vecs, permuted.superbasis_vecs):
                found=True
                data["matrices"].append(cperm_with_mats)
                break
        if not found:
            distinct_superbases[len(distinct_superbases)] = {
                "known_sb": permuted,
                "matrices": [cperm_with_mats]
            }

        assert permuted.is_superbasis()
        assert permuted.is_obtuse(tol=1e-8)
        assert permuted.compute_vonorms().has_same_members(original_vonorms)
        assert permuted.compute_vonorms().conorms.has_same_members(original_conorms)

        manually_permuted_vonorms = original_vonorms.apply_permutation(cperm.to_vonorm_permutation())
        assert manually_permuted_vonorms.about_equal(permuted.compute_vonorms())

        manually_permuted_conorms = original_conorms.apply_permutation(cperm)
        assert manually_permuted_conorms.about_equal(permuted.compute_vonorms().conorms)
        
        tested.append((cperm, cperm_with_mats.matrix))

    return tested

def _assert_all_permutation_matrices_maintain_superbasis(superbasis: Superbasis):
    original_vonorms = superbasis.compute_vonorms()
    original_conorms = original_vonorms.conorms
    tested = []
    distinct_superbases = {}

    for cperm_with_mats in original_conorms.form.permissible_permutations():
        # print(mat, cperm.to_vonorm_permutation())
        for mat in cperm_with_mats.all_matrices:
            cperm = cperm_with_mats.perm
            assert np.linalg.det(mat.matrix) == 1.0
            permuted = superbasis.apply_matrix_transform(mat.matrix)
            # print(f"Testing mat: {permuted.v0()}")
            found = False
            for sb_class_id, data in distinct_superbases.items():
                known_sb = data["known_sb"]
                if superbases_equal(known_sb.superbasis_vecs, permuted.superbasis_vecs):
                    found=True
                    data["matrices"].append(cperm_with_mats)
                    break
            if not found:
                distinct_superbases[len(distinct_superbases)] = {
                    "known_sb": permuted,
                    "matrices": [cperm_with_mats]
                }

            assert permuted.is_superbasis()
            assert permuted.is_obtuse(tol=1e-8)
            assert permuted.compute_vonorms().has_same_members(original_vonorms)
            assert permuted.compute_vonorms().conorms.has_same_members(original_conorms)

            manually_permuted_vonorms = original_vonorms.apply_permutation(cperm.to_vonorm_permutation())
            assert manually_permuted_vonorms.about_equal(permuted.compute_vonorms())

            manually_permuted_conorms = original_conorms.apply_permutation(cperm)
            assert manually_permuted_conorms.about_equal(permuted.compute_vonorms().conorms)
            
            tested.append((cperm, mat))

    return tested


def test_v5_case(reduced_v5_superbasis):
    conorms = reduced_v5_superbasis.compute_vonorms().conorms
    assert len(conorms.form) == 3
    assert conorms.form.voronoi_class == 5
    print(f"{conorms.form.voronoi_class}: {len(conorms.permissible_permutations)}")
    tested = _assert_all_representative_permutation_matrices_maintain_superbasis(reduced_v5_superbasis)

    assert len(tested) == 96

    tested_mats = _assert_all_permutation_matrices_maintain_superbasis(reduced_v5_superbasis)
    assert len(tested_mats) == 384
    assert len(conorms.form.grouped_vonorm_permutations()) == 4


def test_v4_case(reduced_v4_superbasis):
    conorms = reduced_v4_superbasis.compute_vonorms().conorms
    assert len(conorms.form) == 2
    assert conorms.form.voronoi_class == 4
    print(f"{conorms.form.voronoi_class}: {len(conorms.permissible_permutations)}")
    tested = _assert_all_representative_permutation_matrices_maintain_superbasis(reduced_v4_superbasis)
    assert len(tested) == 72
    assert len(conorms.form.grouped_vonorm_permutations()) == 3


def test_v3_case(reduced_v3_superbasis):
    conorms = reduced_v3_superbasis.compute_vonorms().conorms
    assert len(conorms.form) == 2
    assert conorms.form.voronoi_class == 3
    print(f"{conorms.form.voronoi_class}: {len(conorms.permissible_permutations)}")
    tested = _assert_all_representative_permutation_matrices_maintain_superbasis(reduced_v3_superbasis)
    assert len(tested) == 72
    assert len(conorms.form.grouped_vonorm_permutations()) == 3


def test_v2_case(reduced_v2_superbasis: Superbasis):
    conorms = reduced_v2_superbasis.compute_vonorms().conorms
    assert len(conorms.form) == 1
    assert conorms.form.voronoi_class == 2
    print(f"{conorms.form.voronoi_class}: {len(conorms.permissible_permutations)}")
    tested = _assert_all_representative_permutation_matrices_maintain_superbasis(reduced_v2_superbasis)
    assert len(tested) == 48
    print(f"Voronoi Class 2 has: {len(reduced_v2_superbasis.compute_vonorms().conorms.all_permutation_matrices())} unimodular matrices")

    assert len(conorms.form.grouped_vonorm_permutations()) == 2

def test_v1_case(reduced_v1_superbasis):
    conorms: ConormList = reduced_v1_superbasis.compute_vonorms().conorms
    assert len(conorms.form) == 0
    assert conorms.form.voronoi_class == 1
    print(f"{conorms.form.voronoi_class}: {len(conorms.permissible_permutations)}")
    
    mat_tuples = [m.tuple for m in conorms.all_permutation_matrices()]
    print(f"{len(conorms.permissible_permutations)} distinct permutations")
    print(f"{len(set(mat_tuples))} distinct unimodular matrices")
    tested = _assert_all_representative_permutation_matrices_maintain_superbasis(reduced_v1_superbasis)
    assert len(tested) == 24
    assert len(conorms.form.grouped_vonorm_permutations()) == 1


def test_build_all_conorm_lists():
    all_lists = ConormListForm.all_coforms()
    assert len(all_lists) == 38