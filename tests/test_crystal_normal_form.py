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
    print(cnf)