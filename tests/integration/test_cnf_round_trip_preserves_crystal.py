import pytest
from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.lattice.rounding import DiscretizedVonormComputer
from pymatgen.core.structure import Structure
from cnf.unit_cell import UnitCell
import helpers

@helpers.skip_if_fast
@helpers.parameterized_by_mp_struct_idxs(every=50)
@pytest.mark.debug
def test_cnf_round_trip_yields_same_crystal_no_disc(idx, struct: Structure):
    xi = 1
    delta = 100000
    constructor = CNFConstructor(xi, delta, verbose_logging=True)
    uc = UnitCell.from_pymatgen_structure(struct)

    vonorms = uc.vonorms
    motif = uc.motif.discretize(delta)
    cnf = constructor.from_vonorms_and_motif(vonorms, motif).cnf
    recovered_struct = cnf.reconstruct()
    helpers.assert_identical_by_pdd_distance(struct, recovered_struct)

@helpers.skip_if_fast
@helpers.parameterized_by_mp_struct_idxs(every=50)
def test_cnf_round_trip_yields_same_crystal_full_cells(idx, struct: Structure):
    xi = 0.01
    delta = 100000
    constructor = CNFConstructor(xi, delta, verbose_logging=True)
    uc = UnitCell.from_pymatgen_structure(struct)
    # cnf = constructor.from_pymatgen_structure(struct).cnf
    cnf = uc.to_cnf(xi, delta)
    recovered_struct = cnf.reconstruct()
    recovered_two = UnitCell.from_pymatgen_structure(recovered_struct).to_cnf(xi, delta).reconstruct()
    helpers.assert_identical_by_pdd_distance(struct, recovered_struct)

@helpers.skip_if_fast
@helpers.parameterized_by_mp_struct_idxs(every=50)
def test_cnf_round_trip_yields_same_crystal_primitive_cells(idx, struct: Structure):
    xi = 0.01
    delta = 10000

    struct = struct.to_primitive()
    constructor = CNFConstructor(xi, delta)
    cnf = constructor.from_pymatgen_structure(struct).cnf
    recovered_struct = cnf.reconstruct()
    helpers.assert_identical_by_pdd_distance(struct, recovered_struct)
        
@helpers.skip_if_fast
@helpers.parameterized_by_mp_struct_idxs(every=50)
def test_motif_and_superbasis_change_together_with_perms(idx, struct):
    lattice_step_size = 0.00001
    delta = 10000
    constructor = CNFConstructor(lattice_step_size, delta)
    cnf_point = constructor.from_pymatgen_structure(struct).cnf
    helpers.assert_identical_by_pdd_distance(cnf_point.reconstruct(), struct, 0.05)
    perms = cnf_point.lattice_normal_form.vonorms.conorms.form.permissible_permutations()
    dvc = DiscretizedVonormComputer(lattice_step_size)
    for perm in perms:
        # Choose a representative perm - any of these will work
        # because the all_step_vecs chi vectors from David's thesis
        # are already designed to cover S4. So we just need any representative
        # that gets us to each of Kurlin's basis possibilities.

        vonorm_perm = perm.vonorm_permutation

        original_vonorms = cnf_point.lattice_normal_form.vonorms
        original_motif = cnf_point.basis_normal_form.to_motif()
        original_superbasis = original_vonorms.to_superbasis(lattice_step_size)

        assert dvc.find_closest_valid_vonorms(original_superbasis.compute_vonorms()).has_same_members(original_vonorms)

        permuted_vonorms = original_vonorms.apply_permutation(vonorm_perm)

        assert permuted_vonorms.is_obtuse()
        assert permuted_vonorms.is_superbasis()
        assert permuted_vonorms.has_same_members(original_vonorms)
        
        permuted_motif = original_motif.apply_unimodular(perm.matrix)
        
        new_struct = Structure(permuted_vonorms.to_superbasis(lattice_step_size=lattice_step_size).generating_vecs(), permuted_motif.atoms, permuted_motif.positions)
        helpers.assert_identical_by_pdd_distance(struct, new_struct, 0.01)
        
