import pytest
import cnf.lattice.selling as selling
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

    acute_pair = selling.find_first_acute_pair(lattice)
    assert acute_pair[0] == 0
    assert acute_pair[1] == 1

    lattice = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0,0,1],
        [0,1,0.5]
    ])

    acute_pair = selling.find_first_acute_pair(lattice)
    assert acute_pair[0] == 1
    assert acute_pair[1] == 3

def test_selling_transformation():
    lattice = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0,0,0.5],
        [0,1,0.5]
    ])
    transformed = selling.apply_selling_transformation(lattice, 0, 1)

    assert np.all(transformed[0] == [-1, 0, 0])
    assert np.all(transformed[1] == [1, 1, 0])
    assert np.all(transformed[2] == [1, 0, 0.5])
    assert np.all(transformed[3] == [1, 1, 0.5])

    # Apply on indices 2 and 3
    transformed2 = selling.apply_selling_transformation(transformed, 2, 3)

    assert np.all(transformed2[0] == [0, 0, 0.5])
    assert np.all(transformed2[1] == [2, 1, 0.5])
    assert np.all(transformed2[2] == [-1, 0, -0.5])
    assert np.all(transformed2[3] == [1, 1, 0.5])

def test_get_v0():
    lattice = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0,0,0.5]
    ])
    v0 = selling.get_v0_from_generating_vecs(lattice)
    assert np.all(v0 == np.array([-2, -1, -0.5]))