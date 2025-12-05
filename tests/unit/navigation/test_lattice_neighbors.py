import pytest
import numpy as np
import helpers

from cnf.cnf_constructor import CNFConstructor
from cnf import CrystalNormalForm
from cnf.navigation.lattice_neighbor_finder import LatticeStep, LatticeNeighborFinder
from cnf.navigation.neighbor_finder import NeighborFinder

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
    struct = helpers.ALL_MP_STRUCTURES()[0]
    original_cnf = cnf_constructor.from_pymatgen_structure(struct).cnf

    neighbor_set = LatticeNeighborFinder(original_cnf).find_lnf_neighbors()

    for n in neighbor_set.neighbors:
        lnf = n.point
        diff = np.array(sorted(lnf.coords)) - np.array(sorted(original_cnf.lattice_normal_form.coords))
        assert np.sum(np.abs(diff)) == 2
        assert np.max(np.abs(diff)) == 1

def test_tricky_neighbor():
    xi = 1.5
    delta = 5
    element_list = ['Zr', 'Zr']
    cnf = CrystalNormalForm.from_tuple((1, 7, 21, 25, 7, 22, 25, 1, 3, 3), element_list, xi, delta)
    nbs = NeighborFinder(cnf).find_neighbors()


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