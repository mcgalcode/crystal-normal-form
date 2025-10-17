import pytest
import helpers

from pymatgen.core.structure import Structure

from cnf.unit_cell import UnitCell
from cnf.lattice.voronoi import VonormList
from cnf.lattice import Superbasis
from cnf.motif import FractionalMotif
from cnf.lattice.selling import VonormListSellingReducer
from cnf.linalg.unimodular import UNIMODULAR_MATRICES

FREQ=5

@helpers.skip_if_fast
@helpers.parameterized_by_mp_struct_idxs(every=FREQ)
def test_vonorm_stabilizers_preserve_crystal(idx, struct):
    sb = Superbasis.from_pymatgen_structure(struct)
    vn = sb.compute_vonorms()
    reduction_result = VonormListSellingReducer().reduce(vn)
    vn: VonormList = reduction_result.reduced_object
    motif = FractionalMotif.from_pymatgen_structure(struct)
    motif = motif.apply_unimodular(reduction_result.transform_matrix)

    recovered = Structure(vn.to_superbasis().generating_vecs(), motif.atoms, motif.positions)
    helpers.assert_identical_by_pdd_distance(struct, recovered)

    for s in vn.stabilizer_matrices():
        stabilized_vn = vn.to_superbasis(lattice_step_size=1.0).apply_matrix_transform(s.matrix).compute_vonorms()
        assert stabilized_vn.about_equal(vn, tol=5e-3)

        stabilized_motif = motif.apply_unimodular(s)
        recovered = Structure(stabilized_vn.to_superbasis().generating_vecs(), stabilized_motif.atoms, stabilized_motif.positions)
        helpers.assert_identical_by_pdd_distance(struct, recovered)
    
@helpers.skip_if_fast
@helpers.parameterized_by_mp_struct_idxs(every=FREQ)
def test_vonorm_stabilizers_preserve_crystal_motif_only(idx, struct):
    # if len(struct) >= 4:
    #     return
    uc = UnitCell.from_pymatgen_structure(struct).reduce()
    vn = uc.vonorms
    motif = uc.motif
    print("Original MOTIF")
    motif.print_details()
    recovered = uc.to_pymatgen_structure()
    helpers.assert_identical_by_pdd_distance(struct, recovered)

    for s in vn.stabilizer_matrices():
        stabilized_motif = motif.apply_unimodular(s)
        recovered = Structure(vn.to_superbasis().generating_vecs(), stabilized_motif.atoms, stabilized_motif.positions)
        match, reason = helpers.are_structs_geo_matches(struct, recovered, tol=1e-2)
        if not match:
            # struct.to_file("test1.cif")
            # recovered.to_file("test2.cif")
            transform = helpers.get_structure_transformation(struct, recovered)
            # uc.motif.print_details()
            print(f"Found transform, applying to original motif:")
            uc.motif.transform(transform['supercell_matrix']).print_details()
            print(reason)
            print(transform)
            # stabilized_motif.print_details()
        assert match, reason

@helpers.skip_if_fast
@helpers.parameterized_by_mp_struct_idxs(every=FREQ)
def test_vonorm_permutations_preserve_crystal(idx, struct):
    sb = Superbasis.from_pymatgen_structure(struct)
    vn = sb.compute_vonorms()
    reduction_result = VonormListSellingReducer().reduce(vn)
    vn: VonormList = reduction_result.reduced_object
    motif = FractionalMotif.from_pymatgen_structure(struct)
    motif = motif.apply_unimodular(reduction_result.transform_matrix)

    recovered = Structure(vn.to_superbasis().generating_vecs(), motif.atoms, motif.positions)
    helpers.assert_identical_by_pdd_distance(struct, recovered)

    for perm in vn.conorms.permissible_permutations:
        permuted_vn = vn.apply_permutation(perm.vonorm_permutation)
        assert permuted_vn.has_same_members(vn)
        for m in perm.all_matrices:
            permuted_motif = motif.apply_unimodular(m)
            recovered = Structure(permuted_vn.to_superbasis().generating_vecs(), permuted_motif.atoms, permuted_motif.positions)
            helpers.assert_identical_by_pdd_distance(struct, recovered, 0.001)

@helpers.parameterized_by_mp_struct_idxs(every=FREQ)
def test_vonorm_stabilizers_maintain_vonorm_order(idx: int, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()
    # relatively loose tol since these are undiscretize
    tol = 1e-2
    for u in uc.vonorms.stabilizer_matrices():
        uc2 = uc.apply_unimodular(u)
        assert uc2.vonorms.about_equal(uc.vonorms, tol=tol)
        assert uc2.conorms.about_equal(uc.conorms, tol=tol)

@helpers.parameterized_by_mp_struct_idxs(every=FREQ)
def test_vonorm_stabilizer_is_complete(idx: int, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()
    # relatively loose tol since these are undiscretize
    tol = 1e-5
    stab = uc.vonorms.stabilizer_matrices(tol=tol)
    for mat in UNIMODULAR_MATRICES:
        uc2 = uc.apply_unimodular(mat)
        if uc2.vonorms.about_equal(uc.vonorms, tol): 
            if mat not in stab:
                # where is the mat?
                for perm_map in uc.conorms.permissible_permutations:
                    if mat in perm_map.all_matrices:
                        print(f"Found mat in: {perm_map.vonorm_permutation}")
                assert mat not in stab