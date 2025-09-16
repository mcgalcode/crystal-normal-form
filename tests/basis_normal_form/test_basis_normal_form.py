import pytest

from cnf.motif import BasisNormalForm, FractionalMotif
from pymatgen.core.lattice import Lattice

@pytest.fixture
def Zr_HCP_lattice():
    return Lattice.hexagonal(3.19, 1.60 * 3.19)

def test_can_instantiate_from_element_pos_map(sn2_o4_motif: FractionalMotif):
    bnf = BasisNormalForm.from_motif(sn2_o4_motif)

    assert tuple(bnf.coord_list) == (5,5,5,0,3,3,0,7,7,5,2,8,5,8,2)
    assert tuple(bnf.elements) == ("Sn", "Sn", "O","O","O","O")

def test_zr_hcp():
    motif = FractionalMotif.from_elements_and_positions(["Zr", "Zr"], [[0,0,0], [2/3,1/3,1/2]])
    
    print(BasisNormalForm.from_motif(motif, delta=30))

def test_can_round_trip_to_position_map(sn2_o4_motif: FractionalMotif):
    bnf = BasisNormalForm.from_motif(sn2_o4_motif)

    new_pos_map = bnf.to_element_position_map()

    assert tuple(sn2_o4_motif.sorted_elements) == tuple(new_pos_map.sorted_elements)

    for element in sn2_o4_motif.sorted_elements:
        original_positions = {tuple(c) for c in sn2_o4_motif.get_element_positions(element)}
        new_positions = {tuple(c) for c in new_pos_map.get_element_positions(element)}
        assert original_positions == new_positions