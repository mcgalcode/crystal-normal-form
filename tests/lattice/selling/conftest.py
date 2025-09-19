import pytest

from pymatgen.core.lattice import Lattice

@pytest.fixture
def acute_lattice_1():
    return Lattice.from_parameters(1, 2, 3, 25, 35, 60)

@pytest.fixture
def barely_obtuse_lattice():
    return Lattice.from_parameters(1, 2, 3, 85, 80, 88)

@pytest.fixture
def monoclinic_lattice():
    return Lattice.from_parameters(1.5, 1, 2, 30, 50, 130)

@pytest.fixture
def rhombohedral_lattice():
    return Lattice.rhombohedral(1.2, 65)