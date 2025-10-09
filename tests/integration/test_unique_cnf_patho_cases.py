import pytest
import helpers

from cnf.cnf_constructor import CNFConstructor

def test_patho_case_p1():
    structs = helpers.load_pathological_pair("p1")
    helpers.assert_identical_by_pdd_distance(structs[0], structs[1])

    xi = 1.5
    delta = 20

    cnf_builder = CNFConstructor(xi, delta)
    cnf1 = cnf_builder.from_pymatgen_structure(structs[0]).cnf
    cnf2 = cnf_builder.from_pymatgen_structure(structs[1]).cnf
    assert cnf1.lattice_normal_form == cnf2.lattice_normal_form
    assert cnf1 == cnf2