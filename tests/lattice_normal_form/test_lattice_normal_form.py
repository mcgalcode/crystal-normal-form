import pytest
import numpy as np

from pymatgen.core.structure import Lattice
from cnf.lattice_normal_form.lattice_normal_form import LatticeNormalForm
from cnf.lattice_normal_form.selling import get_obtuse_superbasis

@pytest.fixture
def Zr_HCP_lattice():
    return Lattice.hexagonal(3.19, 1.60 * 3.19)

@pytest.fixture
def Zr_BCC_lattice():
    return Lattice.cubic(3.42)

def test_lnf_for_zr_hcp(Zr_HCP_lattice):
    superbasis = get_obtuse_superbasis(Zr_HCP_lattice.matrix)

    permutations, vonorms = LatticeNormalForm.get_canonicalized_superbasis_and_vonorms(superbasis, epsilon=1.5)
    print("Zr HCP lattice vonorm list: ", tuple(vonorms))
    assert tuple(vonorms) == (7,7,17,24,7,24,24) # numbered page 64 of David's thesis
    # print("Vonorms")
    # print(vonorms)
    # print("Permutations")
    # print(permutations)

def test_roundtrip_vonorms_and_back(Zr_BCC_lattice):
    xi = 1.5
    superbasis = get_obtuse_superbasis(Zr_BCC_lattice.matrix)
    # print("First superbasis")
    # print(superbasis)

    # print("Canonicalizng...")
    permutations, vonorms = LatticeNormalForm.get_canonicalized_superbasis_and_vonorms(superbasis, epsilon=xi)
    canonical_generators = LatticeNormalForm.canonical_vonorm_to_canonical_generators(vonorms, xi)
    # print("Found canonical generators:")
    # print(canonical_generators)

    superbasis = get_obtuse_superbasis(canonical_generators)
    # print("Second superbasis")
    # print(superbasis)
    permutations, roundtrip_vonorms = LatticeNormalForm.get_canonicalized_superbasis_and_vonorms(superbasis, epsilon=xi)

    assert tuple(roundtrip_vonorms) == tuple(vonorms)