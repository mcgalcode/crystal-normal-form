from cnf.basis_normal_form import BasisNormalForm

def test_can_instantiate_from_element_pos_map(sn2_o4_el_pos_map):
    bnf = BasisNormalForm.from_element_position_map(sn2_o4_el_pos_map)

    assert tuple(bnf.coord_list) == (5,5,5,0,3,3,0,7,7,5,2,8,5,8,2)
    assert tuple(bnf.elements) == ("Sn", "Sn", "O","O","O","O")


    print(bnf.to_element_position_map())