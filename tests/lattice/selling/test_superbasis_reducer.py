import pytest
from pymatgen.core.structure import Structure

from cnf.lattice.selling.superbasis_reducer import SuperbasisSellingReducer, find_first_acute_pair
from cnf.lattice import Superbasis

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
    transformed = reducer.get_transformed_object(Superbasis(lattice), (0, 1))

    assert np.all(transformed.superbasis_vecs[0] == [-1, 0, 0])
    assert np.all(transformed.superbasis_vecs[1] == [1, 1, 0])
    assert np.all(transformed.superbasis_vecs[2] == [1, 0, 0.5])
    assert np.all(transformed.superbasis_vecs[3] == [1, 1, 0.5])

    # Apply on indices 2 and 3
    transformed2 = reducer.get_transformed_object(transformed, (2, 3))

    assert np.all(transformed2.superbasis_vecs[0] == [0, 0, 0.5])
    assert np.all(transformed2.superbasis_vecs[1] == [2, 1, 0.5])
    assert np.all(transformed2.superbasis_vecs[2] == [-1, 0, -0.5])
    assert np.all(transformed2.superbasis_vecs[3] == [1, 1, 0.5])



def test_can_selling_reduce_superbasis(monoclinic_lattice):
    tol = 1e-7
    reducer = SuperbasisSellingReducer(tol = tol)
    sb = Superbasis.from_pymatgen_lattice(monoclinic_lattice)
    assert not sb.is_obtuse()
    
    reduced = reducer.reduce(sb)
    assert reduced.reduced_object.is_obtuse(tol=tol)

def test_selling_transform_maintains_superbasis(monoclinic_lattice):
    sb = Superbasis.from_pymatgen_lattice(monoclinic_lattice)

    reducer = SuperbasisSellingReducer()

    for i in range(4):
        sb, _ = reducer.apply_selling_transform(sb)
        assert np.isclose(sb.superbasis_vecs[0], -np.sum(sb.superbasis_vecs[1:], axis=0)).all()