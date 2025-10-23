import pytest
import helpers
import numpy as np

from pymatgen.core.structure import Structure
from cnf.unit_cell import UnitCell
from cnf.lattice.voronoi import Coform

from cnf.linalg.unimodular import get_unimodulars_col_max
from cnf.lattice.permutations import UnimodPermMapper

STRUCT_SAMPLE_FREQ = 10

def test_unimodulars_are_det_0():
    for m in UnimodPermMapper.all_unimodular_matrices():
        assert np.isclose(m.determinant(), 1)
        assert m.is_unimodular()

def test_ALL_unimodular_mats_produce_all_possible_coforms():
    handled_vcs = []
    for struct in helpers.ALL_MP_STRUCTURES(1):
        uc = UnitCell.from_pymatgen_structure(struct).reduce()
        voronoi_class = uc.conorms.form.voronoi_class
        if voronoi_class not in handled_vcs:
            print()
            print()
            print(f"Trying struct for class {voronoi_class}")
            matching_coforms = Coform.get_coforms_of_voronoi_class(voronoi_class)
            zero_sets = { cf.zero_indices: [] for cf in matching_coforms }
            for u in get_unimodulars_col_max(4):
                uc2 = uc.apply_unimodular(u)
                uc2.apply_unimodular(u)
                if not uc2.superbasis.is_superbasis():
                    continue

                if not uc2.is_obtuse(tol=1e-8):
                    continue

                if not uc2.vonorms.has_same_members(uc.vonorms):
                    continue

                zero_sets[uc2.conorms.form.zero_indices].append(uc2)
            
            
            satisfied_zeros = [zc for zc, cells in zero_sets.items() if len(cells) > 0]
            unsatisfied_zeros = [zc for zc, cells in zero_sets.items() if len(cells) == 0]

            print(f"Covered {len(satisfied_zeros)} of {len(zero_sets)}...")
            if len(satisfied_zeros) == len(zero_sets):
                print(f"Voronoi class {voronoi_class}")
                for zs, structs in zero_sets.items():
                    print(zs, len(structs))
                    # assert len(structs) > 0
                handled_vcs.append(voronoi_class)
            else:
                print("Couldn't construct representatives for zero sets:")
                for zc in unsatisfied_zeros:
                    print(zc)
    assert len(handled_vcs) == 5

def test_cataloged_unimodular_mats_and_structs_produce_all_possible_coforms():
    handled_vcs = []
    verbose = False
    
    for struct in helpers.ALL_MP_STRUCTURES(10):
        uc = UnitCell.from_pymatgen_structure(struct).reduce()
        voronoi_class = uc.conorms.form.voronoi_class
        if voronoi_class not in handled_vcs:
            helpers.printif("", verbose)
            helpers.printif("", verbose)
            helpers.printif(f"Trying struct for class {voronoi_class}", verbose)
            matching_coforms = Coform.get_coforms_of_voronoi_class(voronoi_class)
            zero_sets = { cf.zero_indices: [] for cf in matching_coforms }
            for perm in uc.conorms.permissible_permutations:
                for u in perm.all_matrices:
                    uc2 = uc.apply_unimodular(u)
                    uc2.apply_unimodular(u)

                    zero_sets[uc2.conorms.form.zero_indices].append(uc2)
                
                
            satisfied_zeros = [zc for zc, cells in zero_sets.items() if len(cells) > 0]
            unsatisfied_zeros = [zc for zc, cells in zero_sets.items() if len(cells) == 0]

            helpers.printif(f"Covered {len(satisfied_zeros)} of {len(zero_sets)}...", verbose)
            if len(satisfied_zeros) == len(zero_sets):
                helpers.printif(f"Voronoi class {voronoi_class}", verbose)
                for zs, structs in zero_sets.items():
                    helpers.printif(zs, verbose)
                    helpers.printif(len(structs), verbose)
                    # assert len(structs) > 0
                handled_vcs.append(voronoi_class)
            else:
                helpers.printif("Couldn't construct representatives for zero sets:", verbose)
                for zc in unsatisfied_zeros:
                    helpers.printif(zc, verbose)
    assert len(handled_vcs) == 5

@helpers.parameterized_by_mp_structs
def test_permissible_perm_mats_maintain_superbasis_and_vonorm_values(idx: int, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()
    tol=1e-2
    for u in uc.conorms.all_permutation_matrices():
        uc2 = uc.apply_unimodular(u)
        assert uc2.is_obtuse(tol=1e-3)
        assert uc2.superbasis.is_superbasis(tol=1e-3)
        assert uc2.vonorms.is_superbasis()
        assert uc2.vonorms.has_same_members(uc.vonorms, tol=tol)
        assert uc2.conorms.has_same_members(uc.conorms, tol=tol)

@helpers.parameterized_by_mp_structs
def test_permissible_perm_mats_maintain_xtal_struct(idx: int, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()
    tol=1e-4
    for u in uc.conorms.all_permutation_matrices():
        uc2 = uc.apply_unimodular(u)
        helpers.assert_identical_by_pdd_distance(struct, uc2.to_pymatgen_structure(), tol)


@helpers.parameterized_by_mp_structs
def test_cataloged_unimodular_matrices_permute_vonorms_correctly(idx, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()
    
    og_perms = uc.vonorms.conorms.form.permissible_permutations()
    catalog_mats = [mat for p in og_perms for mat in p.all_matrices]
    # print()
    # print(f"Struct {idx * STRUCT_SAMPLE_FREQ} has voronoi class: {uc.vonorms.conorms.form.voronoi_class}")
    # print(f"Struct has {len(catalog_mats)} permissible matrices")

    PDD_CUTOFF = 0.0000001
    tol = 1e-2
    for perm in og_perms:
        permuted_vnorms = uc.vonorms.apply_permutation(perm.vonorm_permutation)
        permuted_conorms = uc.vonorms.conorms.apply_permutation(perm.conorm_permutation)
        catalog_mats = perm.all_matrices
        for u in catalog_mats:
            new_uc = uc.apply_unimodular(u)
            assert new_uc.vonorms.about_equal(permuted_vnorms, tol)
            assert new_uc.conorms.about_equal(permuted_conorms, tol)
            pmg_struct = new_uc.to_pymatgen_structure()  
            # if not new_uc.is_obtuse():
            #     print(new_uc.conorms)

            assert new_uc.is_obtuse(tol=tol), f"Non-obtuse result found for struct with vclass {uc.voronoi_class}"
            assert new_uc.superbasis.is_superbasis(tol=tol)
            pdd_dist = helpers.pdd(struct, pmg_struct)
            assert pdd_dist < PDD_CUTOFF


def reps_for_struct(uc: UnitCell):
    target_cfs = [cf.zero_indices for cf in uc.conorms.form.similar_coforms()]
    structs = {}
    for perm_mat in uc.conorms.all_permutation_matrices():
        tuc = uc.apply_unimodular(perm_mat)
        zero_set = tuc.conorms.form.zero_indices
        if zero_set not in structs:
            structs[zero_set] = tuc
        if set(structs.keys()) == set(target_cfs):
            break
    if set(structs.keys()) != set(target_cfs):
        raise RuntimeError("Could not produce all CF representatives for struct!")
    return list(structs.values())

@helpers.parameterized_by_mp_structs
def test_uncataloged_matrices_dont_maintain_lattice(idx: int, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()
    known_perm_mats = uc.conorms.set_tol(1e-4).all_permutation_matrices()
    # print(f"Considerng {len(known_perm_mats)} matrices")
    for umat in get_unimodulars_col_max(2):
        assert umat.determinant() == 1
        uc2 = uc.apply_unimodular(umat)
        fail = False

        if not uc2.is_obtuse(tol=1e-5):
            fail = True
            reason = "Transformed cell was not obtuse"

        if not uc2.superbasis.is_superbasis():
            fail = True
            reason = "superbasis was no longer a superbasis"
        
        if not uc2.vonorms.is_superbasis():
            fail = True
            reason = "vonorms no longer satisfy superbasis invariant"
        
        if not uc2.vonorms.has_same_members(uc.vonorms, tol=1e-4):
            fail = True
            reason = "vonorm values changed"
        
        if not uc2.conorms.has_same_members(uc.conorms, tol=1e-4):
            fail = True
            reason = "conorm values changed"
        
        if umat.tuple == (1, -1, 0, 0, -1, 0, 0, 0, -1):
            print(reason)

        if not fail:
            if umat not in known_perm_mats:
                print()
                print(f"Before conorms: {uc.conorms}")
                print(f"After conorms: {uc2.conorms}")
                print(f"Before vonorms: {uc.vonorms}")
                print(f"After vonorms: {uc2.vonorms}")
                pytest.fail(f"umat {umat.tuple} was not in known permutations, but maintained superbasis for struct at idx {idx * STRUCT_SAMPLE_FREQ} w coform: {uc.conorms.form.zero_indices}")

@helpers.skip_if_fast
@helpers.parameterized_by_mp_structs
def test_uncataloged_matrices_dont_maintain_lattice_comprehensive(idx: int, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()
    reps = reps_for_struct(uc)
    # print(f"Considering {len(reps)} representatives to cover all coforms for struct...")
    for transformed_uc in reps:
        known_perm_mats = transformed_uc.conorms.all_permutation_matrices()
        # print(f"Considerng {len(known_perm_mats)} matrices")
        for umat in get_unimodulars_col_max(2):
            assert umat.determinant() == 1
            uc2 = transformed_uc.apply_unimodular(umat)
            fail = False

            if not uc2.is_obtuse(tol=1e-8):
                fail = True
                reason = "Transformed cell was not obtuse"

            if not uc2.superbasis.is_superbasis():
                fail = True
                reason = "superbasis was no longer a superbasis"
            
            if not uc2.vonorms.is_superbasis():
                fail = True
                reason = "vonorms no longer satisfy superbasis invariant"
            
            if not uc2.vonorms.has_same_members(uc.vonorms, tol=1e-6):
                fail = True
                reason = "vonorm values changed"
            
            if not uc2.conorms.has_same_members(uc.conorms, tol=1e-6):
                fail = True
                reason = "conorm values changed"
            
            if not fail:
                if umat not in known_perm_mats:
                    print()
                    print(f"Before conorms: {transformed_uc.conorms}")
                    print(f"After conorms: {uc2.conorms}")
                    print(f"Before vonorms: {transformed_uc.vonorms}")
                    print(f"After vonorms: {uc2.vonorms}")
                    pytest.fail(f"umat {umat.tuple} was not in known permutations, but maintained superbasis for struct at idx {idx * STRUCT_SAMPLE_FREQ} w coform: {transformed_uc.conorms.form.zero_indices}")

@helpers.parameterized_by_mp_structs
def test_catalog_of_unimodular_matrices_is_complete(idx, struct: Structure):
    verbose=False
    uc = UnitCell.from_pymatgen_structure(struct).reduce()

    original_vonorms = uc.superbasis.compute_vonorms()
    assert original_vonorms.is_obtuse(tol=1e-5)
    assert uc.superbasis.is_obtuse(tol=1e-5)
    tol=1e-5
    
    helpers.printif(original_vonorms.conorms, verbose)
    helpers.printif(original_vonorms.conorms.form.zero_indices, verbose)
    og_perms = original_vonorms.conorms.set_tol(1e-3).form.permissible_permutations()
    og_mats = [mat for p in og_perms for mat in p.all_matrices]
    # helpers.printif(, verbose)
    # helpers.printif(f"Struct {idx * STRUCT_SAMPLE_FREQ} has voronoi class: {original_vonorms.conorms.form.voronoi_class}", verbose)
    # helpers.printif(f"Struct has {len(og_mats)} permissible matrices", verbose)

    PDD_CUTOFF = 0.00001
    equivalent_unit_cells: list[UnitCell] = []
    geo_eq_mats = []
    for u in get_unimodulars_col_max(2):
        new_uc = uc.apply_unimodular(u)
        if not new_uc.is_obtuse(tol=tol) or not new_uc.superbasis.is_superbasis(tol):
            continue

        pmg_struct = new_uc.to_pymatgen_structure()        
        pdd_dist = helpers.pdd(struct, pmg_struct)
        # If the structures are identical
        # And if the original motif does not have inversion symmetry
        # then they should not be mirror images
        if new_uc.vonorms.has_same_members(original_vonorms, tol=tol):
            # if helpers.are_geo_matches(new_uc, uc, tol=1e-3):
            assert pdd_dist < PDD_CUTOFF
            equivalent_unit_cells.append(new_uc)
            geo_eq_mats.append(u)

    if set(geo_eq_mats) != set(og_mats):
        helpers.printif(f"Unequal findings for struct of class {uc.voronoi_class} w form : {uc.conorms.form.zero_indices}", True)
        helpers.printif(uc.conorms, True)
    helpers.printif(f"Found {len(equivalent_unit_cells)} unit cells", verbose)
    assert set(geo_eq_mats).issubset(set(og_mats))