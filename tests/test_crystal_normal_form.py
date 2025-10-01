from cnf.crystal_normal_form import CrystalNormalForm
from cnf.motif import FractionalMotif
from pymatgen.core.lattice import Lattice


def test_crystal_normal_form():
    motif = FractionalMotif.from_elements_and_positions(
        ["Zr", "Zr"],
        [[0, 0, 0], [2/3, 1/3, 1/2]]
    )

    lattice_vecs = Lattice.hexagonal(a=3.19, c=1.6*3.19)
    cnf = CrystalNormalForm.from_motif_and_basis_vecs(motif, lattice_vecs.matrix, motif_step=30)
    assert cnf is not None

def test_crystal_normal_form_equivalence(mp_structures):
    cnf_1 = CrystalNormalForm.from_pymatgen_structure(mp_structures[0])
    cnf_1_1 = CrystalNormalForm.from_pymatgen_structure(mp_structures[0])
    assert cnf_1 == cnf_1_1
    assert len(set([cnf_1, cnf_1_1])) == 1

    cnf_2 = CrystalNormalForm.from_pymatgen_structure(mp_structures[1])
    assert cnf_2 != cnf_1
    assert len(set([cnf_1, cnf_2])) == 2