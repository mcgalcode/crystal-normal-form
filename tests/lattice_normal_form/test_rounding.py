import pytest
import numpy as np

from pymatgen.core.lattice import Lattice
from cnf.lattice_normal_form import DiscretizedVonormComputer
from cnf.lattice.utils import selling_reduce
from cnf.lattice import Superbasis, VonormList
from cnf.lattice_normal_form.lattice_normal_form import LatticeNormalForm

@pytest.fixture
def Zr_BCC_lattice():
    return Lattice.cubic(3.42)

def test_can_discretize_vonorms(Zr_BCC_lattice):
    superbasis = Superbasis.from_pymatgen_lattice(Zr_BCC_lattice)
    vonorms, nsteps = selling_reduce(superbasis.compute_vonorms())
    computer = DiscretizedVonormComputer(vonorms.vonorms, 1.5)
    rounded_vonorms = computer.find_closest_valid_vonorms()

    assert (np.abs(vonorms.vonorms - (rounded_vonorms * 1.5)) < 1.5).all()
