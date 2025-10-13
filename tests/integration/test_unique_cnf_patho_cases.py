import numpy as np
import pytest
import helpers

from cnf.cnf_constructor import CNFConstructor
from cnf.linalg.unimodular import get_unimodulars_col_max, MatrixTuple, load_unimodular
from cnf.motif.utils import move_coords_into_cell
from cnf.motif import FractionalMotif
from cnf.unit_cell import UnitCell

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure, Element

PATHO_IDX = 5


def test_patho_case_mp_395():
    structs = helpers.load_pathological_cifs(f"mp_{PATHO_IDX}")
    helpers.assert_identical_by_pdd_distance(structs[0], structs[1])

    xi = 1.5
    delta = 20

    cnf1 = UnitCell.from_pymatgen_structure(structs[0]).to_cnf(xi, delta)
    cnf2 = UnitCell.from_pymatgen_structure(structs[1]).to_cnf(xi, delta)
    assert cnf1.lattice_normal_form == cnf2.lattice_normal_form
    assert cnf1 == cnf2

    helpers.assert_identical_by_pdd_distance(cnf1.reconstruct(), cnf2.reconstruct())

def test_patho_case_mp_6():
    structs = helpers.load_pathological_cifs("mp_6")
    helpers.assert_identical_by_pdd_distance(structs[0], structs[1])

    xi = 1.5
    delta = 20

    cnf1 = UnitCell.from_pymatgen_structure(structs[0]).to_cnf(xi, delta)
    cnf2 = UnitCell.from_pymatgen_structure(structs[1]).to_cnf(xi, delta)
    assert cnf1 == cnf2
    assert cnf1.lattice_normal_form == cnf2.lattice_normal_form
    assert cnf1 == cnf2

    helpers.assert_identical_by_pdd_distance(cnf1.reconstruct(), cnf2.reconstruct())

def test_patho_case_mp_204():
    structs = helpers.load_pathological_cifs("mp_204")
    helpers.assert_identical_by_pdd_distance(structs[0], structs[1])

    xi = 1.1
    delta = 20

    cnf1 = UnitCell.from_pymatgen_structure(structs[0]).to_cnf(xi, delta)
    cnf2 = UnitCell.from_pymatgen_structure(structs[1]).to_cnf(xi, delta)
    assert cnf1 == cnf2
    assert cnf1.lattice_normal_form == cnf2.lattice_normal_form
    assert cnf1 == cnf2

    helpers.assert_identical_by_pdd_distance(cnf1.reconstruct(), cnf2.reconstruct())

def test_simplify_case_mp_5():
    structs: list[Structure] = helpers.load_pathological_cifs("mp_190_neighbs")

    uc1 = UnitCell.from_pymatgen_structure(structs[0])
    uc2 = UnitCell.from_pymatgen_structure(structs[1])
    helpers.assert_identical_by_pdd_distance(uc1.to_pymatgen_structure(), uc2.to_pymatgen_structure(), cutoff=1e-7)
    
    fm0 = uc1.motif
    fm1 = uc2.motif

    print("S1")
    fm0.print_details()
    print("S2")
    fm1.print_details()
    

    assert not fm0.find_inverted_match(fm1)

    print(f"Original Vonorms s1: {uc1.vonorms}")
    print(f"Original Conorms s1: {uc1.conorms}")
    print()
    print(f"Original Vonorms s2: {uc2.vonorms}")
    print(f"Original Conorms s2: {uc2.conorms}")
    good_combos = []
    bad = []

    # for xi in np.arange(1.0, 2.5, 0.1):
    for xi in [0.3, 1.3, 2.2]:
        print()
        print(f"=============== XI = {xi} =======================")
        xi = round(float(xi), 1)
        # for delta in range(10,20):
        delta = 10
        cnf1 = uc1.to_cnf(xi, delta, verbose=False)
        print()
        print()
        cnf2 = uc2.to_cnf(xi, delta, verbose=False)
        print()
        print(f"Voronoi Class s1: {cnf1.voronoi_class}")
        print(f"Disc Vonorms s1: {cnf1.lattice_normal_form.vonorms}")
        print(f"Disc Conorms s1: {cnf1.lattice_normal_form.vonorms.conorms}")

        print(f"Voronoi Class s2: {cnf2.voronoi_class}")
        print(f"Disc Vonorms s2: {cnf2.lattice_normal_form.vonorms}")
        print(f"Disc Conorms s2: {cnf2.lattice_normal_form.vonorms.conorms}")
        print()
        print(f"CNFs equal: {cnf1 == cnf2}")
        print()
        print(cnf1.coords)
        print(cnf2.coords)
        assert cnf1.lattice_normal_form == cnf2.lattice_normal_form
        if cnf1 == cnf2:
            good_combos.append((xi, delta, cnf1))
        else:
            bad.append((xi, int(delta)))
    

    print(f"Found {len(good_combos)} GOOD combos!")
    print(f"Found {len(bad)} BAD combos!")

    distinctfailingxis = set([pair[0] for pair in bad])
    print(distinctfailingxis)

def test_patho_case_mp_5():
    structs = helpers.load_pathological_cifs("mp_5_doctored")
    uc1 = UnitCell.from_pymatgen_structure(structs[0])
    uc2 = UnitCell.from_pymatgen_structure(structs[1])

    fm0 = uc1.motif
    fm1 = uc2.motif

    # matcher = StructureMatcher(primitive_cell=False)
    # is_match = matcher.fit(structs[0], structs[1])
    # print(f"StructureMatcher says: {is_match}")

    # if is_match:
    #     # Get the transformation
    #     supercell, vector, mapping = matcher.get_transformation(structs[0], structs[1])
    #     print(f"Supercell needed: {supercell}")
    #     print(f"Vector translation needed: {vector}")
    #     print(f"Transformation needed: {mapping}")

    # tfm0 = fm0.transform(supercell)
    # print(fm1.positions)
    # print(tfm0.positions)

    # print()
    # tfm1 = fm1.transform(supercell)
    # print(fm0.positions)
    # print(tfm1.positions)

    

    # helpers.assert_identical_by_pdd_distance(structs[0], structs[1], cutoff=1e-5)
    assert not helpers.are_mirror_images(fm0, fm1)
    xi = 2.0
    delta = 10

    cnf1 = uc1.to_cnf(xi, delta)
    cnf2 = uc2.to_cnf(xi, delta)
    # helpers.assert_identical_by_pdd_distance(structs[0], UnitCell.from_cnf(cnf1).to_pymatgen_structure(), cutoff=1e-5)
    # helpers.assert_identical_by_pdd_distance(structs[1], UnitCell.from_cnf(cnf2).to_pymatgen_structure(), cutoff=1e-5)
    print()
    print("LNF: ", cnf1.lattice_normal_form.coords)
    print("BNF: ", cnf1.basis_normal_form.coord_list)
    print()
    print("LNF: ", cnf2.lattice_normal_form.coords)
    print("BNF: ", cnf2.basis_normal_form.coord_list)
    print()
    print(f"PDD distance was: {helpers.pdd_for_cnfs(cnf1, cnf2)}")
    assert cnf1.lattice_normal_form == cnf2.lattice_normal_form
    assert cnf1 == cnf2
    assert cnf1 == cnf2

    helpers.assert_identical_by_pdd_distance(cnf1.reconstruct(), cnf2.reconstruct())

@pytest.mark.skip
def test_any_unimod_matrices_dont_maintain_xtal():
    structs = helpers.load_pathological_cifs(f"mp_{PATHO_IDX}")
    uc1 = UnitCell.from_pymatgen_structure(structs[0]).reduce()   
    reduced_struct = uc1.to_pymatgen_structure()

    dontmaintain = []
    for m in get_unimodulars_col_max(2):
        tuc1 = uc1.apply_unimodular(m)
        dist = helpers.pdd(reduced_struct, tuc1.to_pymatgen_structure())
        if not dist < 1e-4:
            dontmaintain.append((m, dist))
    
    print(len(dontmaintain))
    print(len(get_unimodulars_col_max(2)))

def test_for_rotoinversion():
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    structs = helpers.load_pathological_cifs("mp_5_simplified")
    helpers.assert_identical_by_pdd_distance(structs[0], structs[1], 1e-5)

    from pymatgen.analysis.structure_matcher import StructureMatcher
    
    matcher = StructureMatcher(
        ltol=1e-5,  # Tight tolerances since we want "genuinely identical"
        stol=1e-5,
        angle_tol=1e-3,
        primitive_cell=False,
        allow_subset=False,
        comparator=None
    )
    # rmsdist, _, transform, _, _ = matcher.get_mapping(structs[0], structs[1])
    
    # print(transform)

    # tuc1 = UnitCell.from_pymatgen_structure(structs[1]).apply_unimodular(MatrixTuple(transform))
    # uc2 = UnitCell.from_pymatgen_structure(structs[0])

    # print(tuc1.superbasis.superbasis_vecs)
    # print()
    # print(uc2.superbasis.superbasis_vecs)
    # print()
    # tuc1.motif.print_details()
    # print()
    # uc2.motif.print_details()
    
    print(matcher.fit(structs[0], structs[1]))

    sg1 = SpacegroupAnalyzer(structs[0])
    sg2 = SpacegroupAnalyzer(structs[1])

    conv1 = sg1.get_conventional_standard_structure()
    conv2 = sg2.get_conventional_standard_structure()

    # Now check if THESE are related by your matrix
    print("Conventional cell 1:")
    print(conv1.lattice.matrix)
    print(conv1.sites)
    print("\nConventional cell 2:")
    print(conv2.lattice.matrix)
    print(conv2.sites)

    # Are the conventional cells identical?
    print(f"\nConventional cells match: {np.allclose(conv1.lattice.matrix, conv2.lattice.matrix)}")

    for i, struct in enumerate([structs[0], structs[1]]):
        sg = SpacegroupAnalyzer(struct, symprec=0.1)
        sym_ops = sg.get_symmetry_operations()
        
        print(f"\nStructure {i+1}:")
        print(f"Space group: {sg.get_space_group_symbol()}")
        
        # Check for inversion symmetry
        has_inversion = any(
            np.allclose(op.rotation_matrix, -np.eye(3)) and 
            np.allclose(op.translation_vector, [0, 0, 0])
            for op in sym_ops
        )
        print(f"Has inversion center: {has_inversion}")
        
        # Count det=-1 operations
        det_neg_count = sum(1 for op in sym_ops if np.linalg.det(op.rotation_matrix) < 0)
        print(f"Number of det=-1 operations: {det_neg_count}")

    matcher_allow_inversion = StructureMatcher(
        primitive_cell=False,
        allow_subset=False
    )

    if matcher_allow_inversion.fit(structs[0], structs[1]):
        mapping = matcher_allow_inversion.get_transformation(structs[0], structs[1])
        supercell, vec, atom_map = mapping
        
        print(f"Determinant of transformation: {np.linalg.det(supercell)}")
        
        if np.linalg.det(supercell) < 0:
            print("🎯 ROTOINVERSION CONFIRMED!")

def test_find_matrix_that_matches():
    print()
    mats_neg_det = load_unimodular("unimodular_6_det_-1.json")
    mats_pos_det = load_unimodular("unimodular_6_det_1.json")
    xi = 0.1
    delta = 10
    structs = helpers.load_pathological_cifs("mp_5_simplified")
    helpers.assert_identical_by_pdd_distance(structs[0], structs[1])
    uc1 = UnitCell.from_pymatgen_structure(structs[0])
    uc2 = UnitCell.from_pymatgen_structure(structs[1])

    dm2 = uc2.motif.discretize(10)

    mot_matches: list[MatrixTuple] = []
    for u in mats_neg_det + mats_pos_det:
        tuc1 = uc1.apply_unimodular(u)
        
        motifs_match = np.all(np.isclose(tuc1.motif.discretize(10).coord_matrix, dm2.coord_matrix, 1e-3))
        sbs_match = np.all(np.isclose(tuc1.superbasis.superbasis_vecs, uc2.superbasis.superbasis_vecs, 1e-3))
        if sbs_match and motifs_match:
            mot_matches.append(u)
    
    print(f"Found {len(mot_matches)} unimod matrices that match!")
    max_norms = {}
    for m in mot_matches:
        max_n = m.col_max_norm()
        curr = max_norms.get(max_n, 0)
        curr = curr + 1
        max_norms[max_n] = curr
    
    for norm, count in max_norms.items():
        print(f"Found {count} mats with col max norm {norm}")

def test_can_patho_pairs_be_transformed_to_eachother():
    print()
    xi = 2.2
    delta = 10
    structs = helpers.load_pathological_cifs("mp_5_simplified")

# Get the space group
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    sg1 = SpacegroupAnalyzer(structs[0])
    sg2 = SpacegroupAnalyzer(structs[1])

    print(f"Structure 1 space group: {sg1.get_space_group_symbol()}")
    print(f"Structure 2 space group: {sg2.get_space_group_symbol()}")

    # Check if they're enantiomorphs (mirror images)
    print(f"Struct 1 chiral: {sg1.get_symmetry_dataset()}")
    print(f"Struct 2 chiral: {sg2.get_symmetry_dataset()}")

    uc1 = UnitCell.from_pymatgen_structure(structs[0])
    uc2 = UnitCell.from_pymatgen_structure(structs[1])
    dm2 = uc2.motif.discretize(delta)

    matches_from_similar_coforms = []
    for u in uc1.conorms.form.all_matrices_for_similar_coforms():
        tuc1 = uc1.apply_unimodular(u)
        helpers.assert_identical_by_pdd_distance(tuc1.to_pymatgen_structure(), uc1.to_pymatgen_structure())
        if np.all(tuc1.motif.discretize(delta).coord_matrix == dm2.coord_matrix):
            matches_from_similar_coforms.append(u)
    
    assert len(matches_from_similar_coforms) > 0
    print("Searched matrices of all coforms that have same voronoi class!")
    print(f"Found {len(matches_from_similar_coforms)} matrices that transform Motif 1 to Motif 2")
    for m in matches_from_similar_coforms:
        print(m.matrix)
    print()

    matches_from_exact_form = []
    for u in uc1.conorms.form.all_matrices():
        tuc1 = uc1.apply_unimodular(u)
        
        if np.all(tuc1.motif.discretize(delta).coord_matrix == dm2.coord_matrix):
            matches_from_exact_form.append(u)
    
    assert len(matches_from_exact_form) > 0
    print("Searched matrices of the exact coform of the original structure")
    print(f"Found {len(matches_from_exact_form)} matrices that transform Motif 1 to Motif 2")
    for m in matches_from_exact_form:
        print(m.matrix)
    print()

    uc1_reduced = uc1.reduce()
    uc2_reduced = uc2.reduce()
    dm2_reduced = uc2_reduced.motif.discretize(delta)

    reduced_matches_similar_coforms = []
    for u in uc1_reduced.conorms.form.all_matrices_for_similar_coforms():
        tuc1 = uc1_reduced.apply_unimodular(u)
        
        if np.all(np.isclose(tuc1.motif.coord_matrix, uc2_reduced.motif.coord_matrix, 1e-5)):
            reduced_matches_similar_coforms.append(u)
    
    assert len(reduced_matches_similar_coforms) > 0
    print("Searched matrices of all coforms that have same voronoi class!")
    print(f"Found {len(matches_from_exact_form)} matrices that transform REDUCED Motif 1 to REDUCED Motif 2")
    for m in matches_from_exact_form:
        print(m.matrix)
    print()

    print("============= TRANSFORMING TO CNF =============")
    print()

    uc1_cnf = uc1.to_cnf(xi, delta)
    print(f"Struct 1 CNF Vonorms: {uc1_cnf.lattice_normal_form.vonorms}")
    uc1_cnf_cell = UnitCell.from_cnf(uc1_cnf)
    uc2_cnf = uc2.to_cnf(xi, delta)
    print(f"Struct 2 CNF Vonorms: {uc2_cnf.lattice_normal_form.vonorms}")
    uc2_cnf_cell = UnitCell.from_cnf(uc2_cnf)

    matches_from_exact_form_after_cnf = []
    for perm_mat in uc1_cnf.lattice_normal_form.vonorms.conorms.permissible_permutations:
        perm = perm_mat.vonorm_permutation
        for u in perm_mat.all_matrices:
            tuc1_cnf = uc1_cnf_cell.apply_unimodular(u)
            
            if np.all(tuc1_cnf.motif.discretize(delta).coord_matrix == uc2_cnf.to_discretized_motif().coord_matrix):
                matches_from_exact_form_after_cnf.append((u, perm))
    
    assert len(matches_from_exact_form_after_cnf) > 0
    print("Searched matrices of the exact CNF coform")
    print(f"Found {len(matches_from_exact_form_after_cnf)} matrices that transform the CNF-Cell Motif 1 to the CNF-Cell Motif 2")
    for m in matches_from_exact_form_after_cnf:
        print(m[1])
        print(m[0].matrix)
    print()

    matches_from_cnf_stabilizer = []
    for u in uc1_cnf.lattice_normal_form.vonorms.stabilizer_matrices():
        tuc1_cnf = uc1_cnf_cell.apply_unimodular(u)
        
        if np.all(tuc1_cnf.motif.discretize(delta).coord_matrix == uc2_cnf.to_discretized_motif().coord_matrix):
            matches_from_cnf_stabilizer.append(u)
    
    # assert len(matches_from_cnf_stabilizer) > 0
    print("Searched matrices in the CNF vonorm stabilizer")
    print(f"Found {len(matches_from_cnf_stabilizer)} matrices that transform the CNF-Cell Motif 1 to the CNF-Cell Motif 2")
    for m in matches_from_cnf_stabilizer:
        print(m.matrix)
    print()

    assert uc1_cnf == uc2_cnf

    cnf1_struct = UnitCell.from_cnf(uc1_cnf).to_pymatgen_structure()
    cnf2_struct = UnitCell.from_cnf(uc2_cnf).to_pymatgen_structure()
    helpers.assert_identical_by_pdd_distance(cnf1_struct, cnf2_struct)

    uc1_cnf_cell = UnitCell.from_cnf(uc1.to_cnf(xi, delta))
    uc2_cnf_cell = UnitCell.from_cnf(uc2.to_cnf(xi, delta))
    dm2_cnf = uc2_cnf_cell.motif.discretize(delta)

    mot_matches = []
    for u in uc1_cnf_cell.conorms.form.all_matrices_for_similar_coforms():
        tuc1 = uc1_cnf_cell.apply_unimodular(u)
        
        if np.all(tuc1.motif.discretize(delta).coord_matrix == dm2_cnf.coord_matrix):
            mot_matches.append(u)
    
    assert len(mot_matches) > 0
    print(mot_matches)


@pytest.mark.skip
def test_debug_positions():
    structs = helpers.load_pathological_cifs(f"mp_{PATHO_IDX}")
    helpers.assert_identical_by_pdd_distance(structs[0], structs[1])

    pos1 = [3, 3, 7]
    pos2 = [13, 13, 17]

    matches = []
    uc1 = UnitCell.from_pymatgen_structure(structs[0])
    for u in get_unimodulars_col_max(2):
        tuc1 = uc1.apply_unimodular(u)
        if not uc1.vonorms.is_superbasis():
            continue

        t1 = move_coords_into_cell(u.inverse() @ pos1, 20)
        # print(t1)
        if np.all(t1 == pos2):
            matches.append(u)

    assert len(matches) > 0
    [print(m) for m in matches]
    xi = 1.5
    delta = 20

    cnf_builder = CNFConstructor(xi, delta, verbose_logging=False)
    cnf1 = cnf_builder.from_pymatgen_structure(structs[1]).cnf
    cnf2 = cnf_builder.from_pymatgen_structure(structs[0]).cnf
    
    # See if this unimodular matrix we found maintains the CNF

    # try just transforming the motif:
    for m in matches:
        tuc1 = uc1.apply_unimodular(m)
        assert uc1.conorms.has_same_members(tuc1.conorms)
        assert uc1.vonorms.has_same_members(tuc1.vonorms)

    t_cnf_1 = UnitCell.from_cnf(cnf1).apply_unimodular(match).to_cnf(xi, delta)
    assert t_cnf_1 == cnf1

    t_cnf_2 = UnitCell.from_cnf(cnf2).apply_unimodular(match).to_cnf(xi, delta)
    assert t_cnf_2 == cnf2

