import pytest

from pymatgen.core.lattice import Lattice
from cnf.lattice_normal_form import DiscretizedVonormComputer
from cnf.lattice_normal_form.selling import get_obtuse_superbasis
from cnf.lattice_normal_form.lattice_normal_form import LatticeNormalForm

@pytest.fixture
def Zr_BCC_lattice():
    return Lattice.cubic(3.42)

def test_can_discretize_vonorms(Zr_BCC_lattice):
    superbasis = get_obtuse_superbasis(Zr_BCC_lattice.matrix)

    vonorms = LatticeNormalForm.compute_vonorms(superbasis)
    print(vonorms)
    computer = DiscretizedVonormComputer(vonorms, 1.5)
    rounded_vonorms = computer.find_closest_valid_vonorms()
    print(rounded_vonorms * 1.5)
