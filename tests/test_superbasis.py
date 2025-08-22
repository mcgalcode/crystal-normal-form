import pytest
from cnf.lattice_normal_form import LatticeNormalForm
from mp_api.client import MPRester
from pymatgen.core.structure import Structure

import numpy as np

@pytest.fixture()
def basic_lattice_1():
    return np.array([
        [1, 0, 0],
        [1, 0.5, 0],
        [0.5, 0.5, 2],
    ])

@pytest.fixture()
def antimony_lattice():
    return Structure.from_file("tests/Sb.cif").lattice

@pytest.fixture()
def zr_hex_lattice():
    return Structure.from_file("tests/Zr_hex.cif").lattice

@pytest.fixture()
def zr_bcc_lattice():
    return Structure.from_file("tests/Zr_bcc.cif").lattice


def test_find_acute_pair():
    lattice = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0,0,0.5],
        [0,1,0.5]
    ])

    acute_pair = LatticeNormalForm.find_first_acute_pair(lattice)
    assert acute_pair[0] == 0
    assert acute_pair[1] == 1

    lattice = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0,0,1],
        [0,1,0.5]
    ])

    acute_pair = LatticeNormalForm.find_first_acute_pair(lattice)
    assert acute_pair[0] == 1
    assert acute_pair[1] == 3

def test_selling_transformation():
    lattice = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0,0,0.5],
        [0,1,0.5]
    ])
    transformed = LatticeNormalForm.apply_selling_transformation(lattice, 0, 1)

    assert np.all(transformed[0] == [-1, 0, 0])
    assert np.all(transformed[1] == [1, 1, 0])
    assert np.all(transformed[2] == [1, 0, 0.5])
    assert np.all(transformed[3] == [1, 1, 0.5])

    # Apply on indices 2 and 3
    transformed2 = LatticeNormalForm.apply_selling_transformation(transformed, 2, 3)

    assert np.all(transformed2[0] == [0, 0, 0.5])
    assert np.all(transformed2[1] == [2, 1, 0.5])
    assert np.all(transformed2[2] == [-1, 0, -0.5])
    assert np.all(transformed2[3] == [1, 1, 0.5])

def test_selling_reduction():
    lattice = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0,0,0.5]
    ])
    obtuse_superbasis = LatticeNormalForm.get_obtuse_superbasis(lattice)
    print(obtuse_superbasis)

    

def test_lattice_normal_form():
    pass

def test_vonorm_permutations(basic_lattice_1):
    
    lattice = basic_lattice_1.T
    obtuse_super_basis = LatticeNormalForm.get_obtuse_superbasis(lattice)
    print(obtuse_super_basis)

    permutations, vonorms = LatticeNormalForm.get_canonicalized_superbasis_and_vonorms(obtuse_super_basis, 0.1)
    print(vonorms)
    print(permutations)

def test_get_dot_products():
    superbasis = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0,0, 1]
    ])

    dots = LatticeNormalForm.get_dot_products(vonorms)

    v0_dot_v1, v0_dot_v2, v0_dot_v3, v1_dot_v2, v1_dot_v3, v2_dot_v3 = LatticeNormalForm.get_dot_products(np.array(canonical_vonorm_string) * discretization)
    print(dots)

def test_get_canonical_generating_vectors():
    vonorms = [2.0, 10.0, 42.0, 45.0, 12.0, 45.0, 42.0]
    basis = LatticeNormalForm.canonical_vonorm_to_canonical_generators(vonorms, 0.1)
    print(basis)

def test_get_v0():
    lattice = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0,0,0.5]
    ])
    v0 = LatticeNormalForm.get_v0_from_generating_vecs(lattice)
    assert np.all(v0 == np.array([-2, -1, -0.5]))


def test_antimony(antimony_lattice):
    print(antimony_lattice)
    superbasis = LatticeNormalForm.get_obtuse_superbasis(antimony_lattice.matrix.T)
    print(superbasis)
    matching_permutations, canonical_vonorm_list = LatticeNormalForm.get_canonicalized_superbasis_and_vonorms(superbasis, 1)
    print(canonical_vonorm_list)

def test_zr_hex_lattice(zr_hex_lattice):
    print(zr_hex_lattice)
    superbasis = LatticeNormalForm.get_obtuse_superbasis(zr_hex_lattice.matrix.T)
    print(superbasis)
    matching_permutations, canonical_vonorm_list = LatticeNormalForm.get_canonicalized_superbasis_and_vonorms(superbasis, 1.5)
    print(canonical_vonorm_list)

def test_zr_bcc_lattice(zr_bcc_lattice):
    print(zr_bcc_lattice)
    superbasis = LatticeNormalForm.get_obtuse_superbasis(zr_bcc_lattice.matrix.T)
    print(superbasis)
    matching_permutations, canonical_vonorm_list = LatticeNormalForm.get_canonicalized_superbasis_and_vonorms(superbasis, 1.5)
    print(canonical_vonorm_list)