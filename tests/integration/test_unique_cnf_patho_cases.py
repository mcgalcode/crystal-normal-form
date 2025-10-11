import numpy as np
import pytest
import helpers

from cnf.cnf_constructor import CNFConstructor
from cnf.linalg.unimodular import UNIMODULAR_MATRICES, UNIMODULAR_MATRICES_MAX_2
from cnf.motif.utils import move_coords_into_cell
from cnf.unit_cell import UnitCell

PATHO_IDX = 6


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

@pytest.mark.skip
def test_any_unimod_matrices_dont_maintain_xtal():
    structs = helpers.load_pathological_cifs(f"mp_{PATHO_IDX}")
    uc1 = UnitCell.from_pymatgen_structure(structs[0]).reduce()   
    reduced_struct = uc1.to_pymatgen_structure()

    dontmaintain = []
    for m in UNIMODULAR_MATRICES_MAX_2:
        tuc1 = uc1.apply_unimodular(m)
        dist = helpers.pdd(reduced_struct, tuc1.to_pymatgen_structure())
        if not dist < 1e-4:
            dontmaintain.append((m, dist))
    
    print(len(dontmaintain))
    print(len(UNIMODULAR_MATRICES_MAX_2))
    print(len(UNIMODULAR_MATRICES))

def test_can_patho_pairs_be_transformed_to_eachother():
    print()
    xi = 1.5
    delta = 20
    structs = helpers.load_pathological_cifs(f"mp_{PATHO_IDX}")
    uc1 = UnitCell.from_pymatgen_structure(structs[0])
    uc2 = UnitCell.from_pymatgen_structure(structs[1])
    dm2 = uc2.motif.discretize(20)

    mot_matches = []
    for u in uc1.conorms.form.all_matrices_for_similar_coforms():
        tuc1 = uc1.apply_unimodular(u)
        
        if np.all(tuc1.motif.discretize(20).coord_matrix == dm2.coord_matrix):
            mot_matches.append(u)
    
    assert len(mot_matches) == 0

    uc1_reduced = uc1.reduce()
    uc2_reduced = uc2.reduce()
    dm2_reduced = uc2_reduced.motif.discretize(20)

    mot_matches = []
    for u in uc1_reduced.conorms.all_permutation_matrices():
        tuc1 = uc1_reduced.apply_unimodular(u)
        
        if np.all(tuc1.motif.discretize(20).coord_matrix == dm2_reduced.coord_matrix):
            mot_matches.append(u)
    
    assert len(mot_matches) > 0

    uc1_cnf = UnitCell.from_cnf(uc1.to_cnf(xi, delta))
    uc2_cnf = UnitCell.from_cnf(uc2.to_cnf(xi, delta))
    dm2_cnf = uc2_cnf.motif.discretize(20)

    mot_matches = []
    for u in uc1_cnf.conorms.form.all_matrices_for_similar_coforms():
        tuc1 = uc1_cnf.apply_unimodular(u)
        
        if np.all(tuc1.motif.discretize(20).coord_matrix == dm2_cnf.coord_matrix):
            mot_matches.append(u)
    
    assert len(mot_matches) > 0
 

@pytest.mark.skip
def test_debug_positions():
    structs = helpers.load_pathological_cifs(f"mp_{PATHO_IDX}")
    helpers.assert_identical_by_pdd_distance(structs[0], structs[1])

    pos1 = [3, 3, 7]
    pos2 = [13, 13, 17]

    matches = []
    uc1 = UnitCell.from_pymatgen_structure(structs[0])
    for u in UNIMODULAR_MATRICES:
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

