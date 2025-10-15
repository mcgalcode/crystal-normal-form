import helpers
from cnf.cnf_constructor import CNFConstructor

@helpers.parameterized_by_mp_structs
def test_repeated_construction_doesnt_change_cnf(idx, struct):
    verbose = True
    xi = 1.5
    delta = 20

    con = CNFConstructor(xi, delta, verbose)

    cnf = con.from_pymatgen_structure(struct).cnf
    for i in range(1):
        new_cnf = con.from_cnf(cnf).cnf
        assert cnf == new_cnf