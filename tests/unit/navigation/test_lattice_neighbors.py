import pytest
import numpy as np
import helpers

from cnf.cnf_constructor import CNFConstructor
from cnf.navigation.lattice_neighbor_finder import LatticeStep, LatticeNeighborFinder

@pytest.fixture()
def cnf_constructor():
    return CNFConstructor(1.5, 10, False)

def test_breaks_if_vec_has_non_one_value():
    vec = [0,0,0,2,0,0,0]
    with pytest.raises(ValueError) as excep:
        LatticeStep(vec, None, None, None)
    
    assert "invalid element != 1" in excep.value.__repr__()

def test_can_find_all_lattice_steps():
    all_steps = LatticeStep.all_step_vecs()
    assert len(all_steps) == 42
    assert len(set([tuple(s) for s in all_steps])) == 42

def test_lattice_neighbor_lnfs_make_sense(cnf_constructor):
    struct = helpers.ALL_MP_STRUCTURES[0]
    original_cnf = cnf_constructor.from_pymatgen_structure(struct).cnf

    neighbor_set = LatticeNeighborFinder(original_cnf).find_lnf_neighbors()

    for n in neighbor_set.neighbors:
        lnf = n.point
        diff = np.array(sorted(lnf.coords)) - np.array(sorted(original_cnf.lattice_normal_form.coords))
        assert np.sum(np.abs(diff)) == 2
        assert np.max(np.abs(diff)) == 1

@helpers.skip_if_fast
@helpers.parameterized_by_mp_struct_idxs(range(0,1000,100))
def test_lattice_neighbor_cnfs_make_sense(idx, struct, cnf_constructor):
    original_cnf = cnf_constructor.from_pymatgen_structure(struct).cnf

    neighbor_set = LatticeNeighborFinder(original_cnf).find_cnf_neighbors()

    for n in neighbor_set.neighbors:
        neighb_lnf = n.point.lattice_normal_form
        diff = np.array(sorted(neighb_lnf.coords)) - np.array(sorted(original_cnf.lattice_normal_form.coords))
        assert np.sum(np.abs(diff)) <= 2
        assert np.max(np.abs(diff)) <= 1