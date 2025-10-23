import pytest
from pymatgen.core.structure import Lattice
from cnf.lattice.lattice_normal_form import LatticeNormalForm
from cnf.lattice.lnf_constructor import LatticeNormalFormConstructor
from cnf.linalg.matrix_tuple import MatrixTuple
from cnf.lattice.permutations import VonormPermutation

@pytest.fixture
def Zr_HCP_lattice():
    return Lattice.hexagonal(3.19, 1.60 * 3.19)

@pytest.fixture
def Zr_BCC_lattice():
    return Lattice.cubic(3.42)

def test_round_trip_to_vonorm_list():
    lnf_constructor = LatticeNormalFormConstructor(1.0)
    test_lattice = Lattice.orthorhombic(1.1, 1.5, 2.0)
    lnf1 = lnf_constructor.build_lnf_from_pymatgen_lattice(test_lattice).lnf
    recovered_superbasis = lnf1.vonorms.to_superbasis()
    lnf2: LatticeNormalForm = lnf_constructor.build_lnf_from_superbasis(recovered_superbasis).lnf
    assert lnf1 == lnf2

def test_can_round_trip_trivial_discretization():
    lnf_constructor = LatticeNormalFormConstructor(1.0)
    test_lattice = Lattice.orthorhombic(1.1, 1.5, 2.0)
    lnf1: LatticeNormalForm = lnf_constructor.build_lnf_from_pymatgen_lattice(test_lattice).lnf
    lnf2: LatticeNormalForm = lnf_constructor.build_lnf_from_superbasis(lnf1.to_superbasis()).lnf
    lnf3: LatticeNormalForm = lnf_constructor.build_lnf_from_superbasis(lnf2.to_superbasis()).lnf

    assert lnf2.to_superbasis() == lnf3.to_superbasis()
    assert lnf1.to_superbasis() == lnf3.to_superbasis()

def test_lnf_for_zr_hcp(Zr_HCP_lattice):
    lnf_constructor = LatticeNormalFormConstructor(1.5)
    lnf = lnf_constructor.build_lnf_from_pymatgen_lattice(Zr_HCP_lattice).lnf
    # print("Zr HCP lattice vonorm list: ", lnf.vonorms)
    assert lnf.vonorms.tuple == (7,7,17,24,7,24,24) # numbered page 64 of David's thesis

