import pytest
import helpers

from pymatgen.core.structure import Structure

from cnf.lattice.voronoi import VonormList
from cnf.lattice import Superbasis
from cnf.motif import FractionalMotif
from cnf.lattice.selling import VonormListSellingReducer

@helpers.skip_if_fast
@helpers.parameterized_by_mp_structs
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
        assert stabilized_vn == vn

        stabilized_motif = motif.apply_unimodular(s)
        recovered = Structure(stabilized_vn.to_superbasis().generating_vecs(), stabilized_motif.atoms, stabilized_motif.positions)
        helpers.assert_identical_by_pdd_distance(struct, recovered)

@helpers.skip_if_fast
@helpers.parameterized_by_mp_structs
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

        permuted_motif = motif.apply_unimodular(perm.matrix)
        recovered = Structure(permuted_vn.to_superbasis().generating_vecs(), permuted_motif.atoms, permuted_motif.positions)
        helpers.assert_identical_by_pdd_distance(struct, recovered, 0.001)
