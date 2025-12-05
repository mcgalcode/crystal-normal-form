import pytest
import numpy as np
import helpers

from cnf.cnf_constructor import CNFConstructor
from cnf import CrystalNormalForm
from cnf.navigation.neighbor_finder import NeighborFinder

@pytest.fixture()
def cnf_constructor():
    return CNFConstructor(1.5, 10, False)

@helpers.parameterized_by_mp_struct_idxs([10])
def test_cnf_neighbors_are_unique(idx, struct, cnf_constructor):
    original_cnf = cnf_constructor.from_pymatgen_structure(struct).cnf

    neighbs = NeighborFinder(original_cnf).find_neighbors()

    assert len(set(neighbs)) == len(neighbs)

@helpers.parameterized_by_mp_struct_idxs([10])
def test_self_is_not_neighbor(idx, struct, cnf_constructor):
    original_cnf = cnf_constructor.from_pymatgen_structure(struct).cnf

    neighbs = NeighborFinder(original_cnf).find_neighbors()

    assert original_cnf not in set(neighbs)

@helpers.parameterized_by_mp_struct_idxs([10])
def test_neighbors_are_reciproical(idx, struct, cnf_constructor):
    original_cnf = cnf_constructor.from_pymatgen_structure(struct).cnf

    neighbs = NeighborFinder(original_cnf).find_neighbors()

    for n in neighbs:
        n2s = NeighborFinder(n).find_neighbors()
        assert original_cnf in n2s

@helpers.parameterized_by_mp_struct_idxs([10])
def test_lattice_neighbs_neighbors_are_close(idx, struct, cnf_constructor):

    original_cnf = cnf_constructor.from_pymatgen_structure(struct).cnf
    # helpers.printif(f"Original CNF: {original_cnf.coords}", verbose)
    neigb_set = NeighborFinder(original_cnf).find_neighbors()
    for n in neigb_set:
        pdd = helpers.assertions.pdd_for_cnfs(n, original_cnf, k=100)
        exact_geo_matches, reason = helpers.are_cnfs_geo_matches(n, original_cnf)
        assert pdd < (cnf_constructor.xi / 2) and not exact_geo_matches, reason