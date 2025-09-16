import pytest
import numpy as np

from pymatgen.core.structure import Lattice
from cnf.lattice.lattice_normal_form import LatticeNormalForm
from cnf.lattice import Superbasis
from cnf.lattice.utils import selling_reduce

@pytest.fixture
def Zr_HCP_lattice():
    return Lattice.hexagonal(3.19, 1.60 * 3.19)

@pytest.fixture
def Zr_BCC_lattice():
    return Lattice.cubic(3.42)

def test_round_trip_to_vonorm_list():
    test_lattice = Lattice.orthorhombic(1.1, 1.5, 2.0)
    lnf1: LatticeNormalForm = LatticeNormalForm.from_pymatgen_lattice(test_lattice, lattice_step_size=1.0, return_transforms=False)
    recovered_superbasis = lnf1.vonorms.to_superbasis()
    lnf2: LatticeNormalForm = LatticeNormalForm.from_superbasis(recovered_superbasis, lattice_step_size=1.0, return_transforms=False)
    assert lnf1 == lnf2

def test_can_round_trip_trivial_discretization():
    test_lattice = Lattice.orthorhombic(1.1, 1.5, 2.0)
    lnf1: LatticeNormalForm = LatticeNormalForm.from_pymatgen_lattice(test_lattice, lattice_step_size=1.0, return_transforms=False)
    lnf2: LatticeNormalForm = LatticeNormalForm.from_superbasis(lnf1.to_superbasis(), lattice_step_size=1.0, return_transforms=False)
    lnf3: LatticeNormalForm = LatticeNormalForm.from_superbasis(lnf2.to_superbasis(), lattice_step_size=1.0, return_transforms=False)

    print(lnf2.to_superbasis())
    print(lnf3.to_superbasis())

def test_lnf_for_zr_hcp(Zr_HCP_lattice):

    lnf = LatticeNormalForm.from_pymatgen_lattice(Zr_HCP_lattice, lattice_step_size=1.5, return_transforms=False)
    # print("Zr HCP lattice vonorm list: ", lnf.vonorms)
    assert lnf.vonorms.vonorms == (7,7,17,24,7,24,24) # numbered page 64 of David's thesis