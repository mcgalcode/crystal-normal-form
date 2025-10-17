import pytest
import helpers
from cnf.cnf_constructor import CNFConstructor

@helpers.parameterized_by_mp_struct_idxs(every=50)
def test_repeated_construction_doesnt_change_cnf(idx, struct):
    verbose = True
    xi = 1.5
    delta = 20

    con = CNFConstructor(xi, delta, verbose)
    cnf = con.from_pymatgen_structure(struct).cnf
    for _ in range(5):
        new_cnf = con.from_cnf(cnf).cnf
        assert new_cnf == cnf
        cnf = new_cnf