from cnf.lattice import Superbasis
from cnf.lattice.permutations import ConormPermutation, VonormPermutation
from cnf.lattice.voronoi import ConormListForm, ConormList

def _assert_all_permutation_matrices_maintain_superbasis(superbasis: Superbasis):
    original_vonorms = superbasis.compute_vonorms()
    original_conorms = original_vonorms.conorms
    tested = []
    for cperm_with_mats in original_conorms.form.permissible_permutations():
        # print(mat, cperm.to_vonorm_permutation())
        cperm = cperm_with_mats.perm
        permuted = superbasis.apply_matrix_transform(cperm_with_mats.matrix.matrix)
        # print(f"Testing mat: {permuted.v0()}")
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

def test_v5_case(reduced_v5_superbasis):
    conorms = reduced_v5_superbasis.compute_vonorms().conorms
    assert len(conorms.form) == 3
    assert conorms.form.voronoi_class == 5
    print(f"{conorms.form.voronoi_class}: {len(conorms.permissible_permutations)}")
    tested = _assert_all_permutation_matrices_maintain_superbasis(reduced_v5_superbasis)

    assert len(tested) == 96


def test_v4_case(reduced_v4_superbasis):
    conorms = reduced_v4_superbasis.compute_vonorms().conorms
    assert len(conorms.form) == 2
    assert conorms.form.voronoi_class == 4
    print(f"{conorms.form.voronoi_class}: {len(conorms.permissible_permutations)}")
    tested = _assert_all_permutation_matrices_maintain_superbasis(reduced_v4_superbasis)
    assert len(tested) == 72

def test_v3_case(reduced_v3_superbasis):
    conorms = reduced_v3_superbasis.compute_vonorms().conorms
    assert len(conorms.form) == 2
    assert conorms.form.voronoi_class == 3
    print(f"{conorms.form.voronoi_class}: {len(conorms.permissible_permutations)}")
    tested = _assert_all_permutation_matrices_maintain_superbasis(reduced_v3_superbasis)
    assert len(tested) == 72

def test_v2_case(reduced_v2_superbasis):
    conorms = reduced_v2_superbasis.compute_vonorms().conorms
    assert len(conorms.form) == 1
    assert conorms.form.voronoi_class == 2
    print(f"{conorms.form.voronoi_class}: {len(conorms.permissible_permutations)}")
    tested = _assert_all_permutation_matrices_maintain_superbasis(reduced_v2_superbasis)
    assert len(tested) == 48

def test_v1_case(reduced_v1_superbasis):
    conorms: ConormList = reduced_v1_superbasis.compute_vonorms().conorms
    assert len(conorms.form) == 0
    assert conorms.form.voronoi_class == 1
    print(f"{conorms.form.voronoi_class}: {len(conorms.permissible_permutations)}")

    unimodular_matrices = []
    for p in conorms.permissible_permutations:
        unimodular_matrices = unimodular_matrices + [p.matrix]
    
    mat_tuples = [m.tuple for m in unimodular_matrices]
    print(f"{len(conorms.permissible_permutations)} distinct permutations")
    print(f"{len(set(mat_tuples))} distinct unimodular matrices")
    tested = _assert_all_permutation_matrices_maintain_superbasis(reduced_v1_superbasis)
    assert len(tested) == 24

def test_build_all_conorm_lists():
    all_lists = ConormListForm.all_coforms()
    assert len(all_lists) == 42