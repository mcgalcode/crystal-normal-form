import pytest
import helpers
import numpy as np

from cnf.motif import FractionalMotif, MNFConstructor
from cnf.linalg import MatrixTuple
import cnf.motif.mnf_constructor as mnfc
from cnf.linalg.unimodular import get_unimodulars_col_max
from pymatgen.core.lattice import Lattice

@pytest.fixture
def Zr_HCP_lattice():
    return Lattice.hexagonal(3.19, 1.60 * 3.19)

def test_can_instantiate_from_element_pos_map(sn2_o4_motif: FractionalMotif):
    sn2_o4_motif = sn2_o4_motif.discretize(10)
    constructor = MNFConstructor(10)
    mnf_result = constructor.build(sn2_o4_motif)
    mnf = mnf_result.mnf

    assert tuple(mnf.coord_list) == (5,5,5,0,3,3,0,7,7,5,2,8,5,8,2)
    assert tuple(mnf.elements) == ("Sn", "Sn", "O","O","O","O")

def test_zr_hcp():
    motif = FractionalMotif.from_elements_and_positions(["Zr", "Zr"], [[0,0,0], [2/3,1/3,1/2]])
    motif = motif.discretize(30)
    constructor = MNFConstructor(30)
    mnf = constructor.build(motif).mnf
    assert mnf.coord_list == (10,20,15)

def test_can_round_trip_to_position_map(sn2_o4_motif: FractionalMotif):
    mnf = MNFConstructor(10).build(sn2_o4_motif.discretize(10)).mnf

    new_motif = mnf.to_motif()

    assert tuple(sn2_o4_motif.sorted_elements) == tuple(new_motif.sorted_elements)

    for element in sn2_o4_motif.sorted_elements:
        original_positions = {tuple(c) for c in sn2_o4_motif.get_element_positions(element)}
        new_positions = {tuple(c) for c in new_motif.get_element_positions(element)}
        assert original_positions == new_positions

def test_can_get_inverses():
    stab_mats = get_unimodulars_col_max(2)[:19]
    inverses = [m.inverse() for m in stab_mats]

    np_mats = [m.matrix for m in stab_mats]
    computed_inverses = mnfc.invert_unimods(np_mats)
    computed_inverses = [MatrixTuple(i) for i in computed_inverses]

    assert set(computed_inverses) == set(inverses)

def assert_mnf_tup_sets_eq(mnfs1, mnfs2):
    assert len(mnfs1) == len(mnfs2)
    def _proc_float(f):
        return round(float(f),6)    
    mnfs1 = [tuple([_proc_float(i) for i in l]) for l in mnfs1]
    mnfs2 = [tuple([_proc_float(i) for i in l]) for l in mnfs2]
    assert set(mnfs1) == set(mnfs2)


def test_can_convert_coord_mats_to_mnf_lists():
    structs = helpers._ALL_MP_STRUCTURES[:10]
    motifs = [FractionalMotif.from_pymatgen_structure(s) for s in structs]


    coord_mats = [m.coord_matrix for m in motifs]
    mnfs1 = mnfc.get_mnf_strs_from_coord_mats(coord_mats)
    mnfs2 = [m.to_mnf_list() for m in motifs]
    assert_mnf_tup_sets_eq(mnfs1, mnfs2)


@helpers.parameterized_by_mp_struct_idxs([0])
def test_vectorized_stabilizers(idx, struct):
    motif = FractionalMotif.from_pymatgen_structure(struct)
    stab_mats = get_unimodulars_col_max(2)[:2]


    motif_coord_mats = mnfc.get_stabilized_coord_mats(np.array([m.matrix for m in stab_mats]), motif)
    motif_coord_mats = mnfc.move_coords_into_bounds(motif_coord_mats, motif._mod)

    assert len(motif_coord_mats) == len(stab_mats)

    manual_mnfs = []
    for m in stab_mats:
        t = motif.apply_unimodular(m)
        manual_mnfs.append(t.to_mnf_list())
    
    mnfs = mnfc.get_mnf_strs_from_coord_mats(motif_coord_mats)
    assert_mnf_tup_sets_eq(mnfs, manual_mnfs)

@helpers.parameterized_by_mp_struct_idxs([3])
def test_get_shifted_coord_mats(idx, struct):
    motif = FractionalMotif.from_pymatgen_structure(struct)
    assert motif.num_origin_atoms > 2

    shifted_cms = mnfc.get_all_shifted_coord_mats(motif.coord_matrix, motif.num_origin_atoms, motif._mod)
    shifted_mnfs = mnfc.get_mnf_strs_from_coord_mats(shifted_cms)

    shifted_motifs, shifts = mnfc.get_all_shifted_motifs(motif)
    motif_mnfs = [sm.to_mnf_list() for sm in shifted_motifs]
    assert_mnf_tup_sets_eq(shifted_mnfs, motif_mnfs)

@helpers.parameterized_by_mp_struct_idxs([3])
def test_can_sort_motif_coord_matrix(idx, struct):
    motif = FractionalMotif.from_pymatgen_structure(struct)
    print(motif.atoms)
    assert motif.num_origin_atoms > 2

    expected_bnf_list = motif.to_mnf_list(sort=True)

    atom_labs = mnfc.get_atom_labels(motif)
    sorted_motif_coord_mat = mnfc.sort_motif_coord_arr(motif.coord_matrix, atom_labs)
    sorted_mnf_list = mnfc.get_mnf_strs_from_coord_mats([sorted_motif_coord_mat])
    assert_mnf_tup_sets_eq([expected_bnf_list], sorted_mnf_list)
