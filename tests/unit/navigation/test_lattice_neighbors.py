import pytest
import numpy as np
import helpers

from cnf.cnf_constructor import CNFConstructor
from cnf import CrystalNormalForm
from cnf.navigation.lattice_neighbor_finder import all_step_vecs, LatticeNeighborFinder
from cnf.navigation.neighbor_finder import NeighborFinder

@pytest.fixture()
def cnf_constructor():
    return CNFConstructor(1.5, 10, False)

def test_can_find_all_lattice_steps():
    steps = all_step_vecs()
    assert len(steps) == 42
    assert len(set([tuple(s) for s in steps])) == 42

@helpers.parameterized_by_mp_struct_idxs([537])  # mp-1977794
def test_lattice_neighbor_lnfs_make_sense(idx, struct, cnf_constructor):
    original_cnf = cnf_constructor.from_pymatgen_structure(struct).cnf

    neighbor_set = NeighborFinder.from_cnf(original_cnf).find_lattice_neighbor_cnfs(original_cnf)

    for n in neighbor_set:
        lnf = n.lattice_normal_form
        diff = np.array(sorted(lnf.coords)) - np.array(sorted(original_cnf.lattice_normal_form.coords))
        assert np.sum(np.abs(diff)) == 2
        assert np.max(np.abs(diff)) == 1


@helpers.skip_if_fast
@helpers.parameterized_by_mp_struct_idxs(range(0,1000,100))
def test_lattice_neighbor_cnfs_make_sense(idx, struct, cnf_constructor):
    original_cnf = cnf_constructor.from_pymatgen_structure(struct).cnf

    neighbor_set = NeighborFinder.from_cnf(original_cnf).find_lattice_neighbor_cnfs(original_cnf)

    for n in neighbor_set:
        neighb_lnf = n.lattice_normal_form
        diff = np.array(sorted(neighb_lnf.coords)) - np.array(sorted(original_cnf.lattice_normal_form.coords))
        assert np.sum(np.abs(diff)) <= 2
        assert np.max(np.abs(diff)) <= 1