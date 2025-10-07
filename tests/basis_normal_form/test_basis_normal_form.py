import pytest

from cnf.motif import FractionalMotif, BNFConstructor
from pymatgen.core.lattice import Lattice

@pytest.fixture
def Zr_HCP_lattice():
    return Lattice.hexagonal(3.19, 1.60 * 3.19)

def test_can_instantiate_from_element_pos_map(sn2_o4_motif: FractionalMotif):
    constructor = BNFConstructor()
    bnf_result = constructor.build_from_fractional_motif(sn2_o4_motif)
    bnf = bnf_result.bnf

    assert tuple(bnf.coord_list) == (5,5,5,0,3,3,0,7,7,5,2,8,5,8,2)
    assert tuple(bnf.elements) == ("Sn", "Sn", "O","O","O","O")

def test_zr_hcp():
    motif = FractionalMotif.from_elements_and_positions(["Zr", "Zr"], [[0,0,0], [2/3,1/3,1/2]])
    
    constructor = BNFConstructor()
    bnf = constructor.build_from_fractional_motif(motif, delta=30).bnf
    assert bnf.coord_list == (10,20,15)

def test_can_round_trip_to_position_map(sn2_o4_motif: FractionalMotif):
    bnf = BNFConstructor().build_from_fractional_motif(sn2_o4_motif).bnf

    new_motif = bnf.to_motif()

    assert tuple(sn2_o4_motif.sorted_elements) == tuple(new_motif.sorted_elements)

    for element in sn2_o4_motif.sorted_elements:
        original_positions = {tuple(c) for c in sn2_o4_motif.get_element_positions(element)}
        new_positions = {tuple(c) for c in new_motif.get_element_positions(element)}
        assert original_positions == new_positions