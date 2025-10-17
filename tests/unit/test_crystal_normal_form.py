import helpers

from cnf.cnf_constructor import CNFConstructor
from cnf.motif import FractionalMotif
from pymatgen.core.lattice import Lattice


def test_crystal_normal_form():
    motif = FractionalMotif.from_elements_and_positions(
        ["Zr", "Zr"],
        [[0, 0, 0], [2/3, 1/3, 1/2]]
    )

    lattice_vecs = Lattice.hexagonal(a=3.19, c=1.6*3.19)
    constructor = CNFConstructor(
        xi=0.15,
        delta=30
    )
    cnf = constructor.from_motif_and_basis_vecs(motif, lattice_vecs.matrix).cnf
    assert cnf is not None

def test_crystal_normal_form_equivalence():
    constructor = CNFConstructor(
        xi=0.15,
        delta=30
    )
    cnf_1 = constructor.from_pymatgen_structure(helpers.ALL_MP_STRUCTURES()[0]).cnf
    cnf_1_1 = constructor.from_pymatgen_structure(helpers.ALL_MP_STRUCTURES()[0]).cnf
    assert cnf_1 == cnf_1_1
    assert len(set([cnf_1, cnf_1_1])) == 1

    cnf_2 = constructor.from_pymatgen_structure(helpers.ALL_MP_STRUCTURES()[1]).cnf
    assert cnf_2 != cnf_1
    assert len(set([cnf_1, cnf_2])) == 2