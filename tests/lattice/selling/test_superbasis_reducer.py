import pytest
from pymatgen.core.structure import Structure

from cnf.lattice.selling.superbasis_reducer import SuperbasisSellingReducer, find_first_acute_pair

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

    acute_pair = find_first_acute_pair(lattice)
    assert acute_pair[0] == 0
    assert acute_pair[1] == 1

    lattice = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0,0,1],
        [0,1,0.5]
    ])

    acute_pair = find_first_acute_pair(lattice)
    assert acute_pair[0] == 1
    assert acute_pair[1] == 3

def test_selling_transformation():
    lattice = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0,0,0.5],
        [0,1,0.5]
    ])
    reducer = SuperbasisSellingReducer()
    transformed = reducer.get_transformed_vecs(lattice, 0, 1)

    assert np.all(transformed[0] == [-1, 0, 0])
    assert np.all(transformed[1] == [1, 1, 0])
    assert np.all(transformed[2] == [1, 0, 0.5])
    assert np.all(transformed[3] == [1, 1, 0.5])

    # Apply on indices 2 and 3
    transformed2 = reducer.get_transformed_vecs(transformed, 2, 3)

    assert np.all(transformed2[0] == [0, 0, 0.5])
    assert np.all(transformed2[1] == [2, 1, 0.5])
    assert np.all(transformed2[2] == [-1, 0, -0.5])
    assert np.all(transformed2[3] == [1, 1, 0.5])