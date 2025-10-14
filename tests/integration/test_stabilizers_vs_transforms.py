import helpers
import pytest
import numpy as np
from cnf.lattice.permutations import is_permutation_set_closed
from cnf.lattice.lnf_constructor import VonormSorter
from cnf.unit_cell import UnitCell
from cnf.linalg import MatrixTuple
from cnf.motif.atomic_motif import DiscretizedMotif
from cnf.motif.bnf_constructor import get_all_shifted_motifs

@helpers.parameterized_by_mp_structs
def test_stabilizers_form_groups(idx, struct):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()
    eye = (0,1,2,3,4,5,6)
    for p in uc.conorms.permissible_permutations:
        vonorms = uc.vonorms.apply_permutation(p.vonorm_permutation)
        vo_stab = [p.vonorm_permutation for p in vonorms.stabilizer_perms(tol=1e-4)]
        # print(vo_stab)
        assert is_permutation_set_closed(vo_stab)
        assert eye in vo_stab

        co_stab = [p.conorm_permutation for p in vonorms.stabilizer_perms(tol=1e-4)]
        assert is_permutation_set_closed(co_stab)
        assert eye in co_stab

        mats = vonorms.stabilizer_matrices(tol=1e-4)
        # print(len(mats))
        # if len(vo_stab) != len(mats):
        #     print(f"{len(vo_stab)} Permutaions")
        #     print(f"{len(mats)} Matrices")
        #     print()
        for mat1 in mats:
            for mat2 in mats:
                # assert mat1 @ MatrixTuple(np.random.randint(1,3, size=(3,3))) in mats
                assert mat1 @ mat2 in mats
        
        for mat in mats:
            assert mat.inverse() in mats

def test_transforms_for_sort_equivalent_to_stabilizer():
    fail_structs = []
    for struct in helpers.ALL_MP_STRUCTURES():
        uc = UnitCell.from_pymatgen_structure(struct).reduce()

        sorted_vonorms, eq_transforms = VonormSorter().get_canonicalized_vonorms(uc.vonorms)

        if not len(sorted_vonorms.stabilizer_perms(tol=1e-3)) == len(eq_transforms):
            fail_structs.append(struct)
    
    assert len(fail_structs) < 0.01 * len(helpers.ALL_MP_STRUCTURES())


@helpers.parameterized_by_mp_structs
def test_motif_is_stabilized_by_vonorm_stabilizers(idx, struct):
    delta = 20
    uc = UnitCell.from_pymatgen_structure(struct).reduce()
    for p in uc.conorms.permissible_permutations:
        vonorms = uc.vonorms.apply_permutation(p.vonorm_permutation)
        stabs = vonorms.stabilizer_matrices(tol=1e-4)

        disc_motif = uc.motif.discretize(delta)

        first_round_transformed_motifs: list[DiscretizedMotif] = []
        for stab in stabs:
            first_round_transformed_motifs.append(disc_motif.apply_unimodular(stab))

        first_round_positions = [tuple(m.position_tuple_list) for m in first_round_transformed_motifs]        

        for m in first_round_transformed_motifs:
            for stab in stabs:
                m2 = m.apply_unimodular(stab)
                assert tuple(m2.position_tuple_list) in first_round_positions

@helpers.parameterized_by_mp_structs
def test_motif_is_stabilized_wrt_mats_and_origin_shifts(idx, struct):
    delta = 20
    uc = UnitCell.from_pymatgen_structure(struct).reduce()
    vonorms = uc.vonorms
    stabs = vonorms.stabilizer_matrices(tol=1e-4)

    disc_motif = uc.motif.discretize(delta)


    first_round_transformed_motifs: list[DiscretizedMotif] = []
    for stab in stabs:
        motifs, shifts = get_all_shifted_motifs(disc_motif.apply_unimodular(stab))
        first_round_transformed_motifs.extend(motifs)

    first_round_positions = [tuple(m.position_tuple_list) for m in first_round_transformed_motifs]        

    for m in first_round_transformed_motifs:
        for stab in stabs:
            m2 = m.apply_unimodular(stab)
            shifted_ms, shifts = get_all_shifted_motifs(m2)
            for m3 in shifted_ms:
                assert tuple(m3.position_tuple_list) in first_round_positions