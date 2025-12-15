import pytest
from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.lattice.rounding import DiscretizedVonormComputer
from pymatgen.core.structure import Structure
from cnf.unit_cell import UnitCell
import helpers


@helpers.parameterized_by_mp_structs
def test_cnf_round_trip_yields_same_crystal_no_disc(idx, struct: Structure):
    xi = 1
    delta = 10
    constructor = CNFConstructor(xi, delta, verbose_logging=True)
    uc = UnitCell.from_pymatgen_structure(struct)
    cnf = uc.to_cnf(xi, delta)
    assert set(cnf.lattice_normal_form.vonorms.stabilizer_matrices()) == set(cnf.lattice_normal_form.vonorms.stabilizer_matrices_fast())
