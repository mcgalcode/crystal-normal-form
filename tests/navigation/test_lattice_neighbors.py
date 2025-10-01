import pytest

from cnf.navigation.lattice_neighbors import LatticeStep

def test_breaks_if_vec_has_non_one_value():
    vec = [0,0,0,2,0,0,0]
    with pytest.raises(ValueError) as excep:
        LatticeStep(vec)
    
    assert "invalid element != 1" in excep.value.__repr__()

def test_breaks_if_vec_has_imbalanced_values():
    vec = [0,-1,0,-1,0,0,0]
    with pytest.raises(ValueError) as excep:
        LatticeStep(vec)
    
    assert "imbalanced primary" in excep.value.__repr__()

def test_can_find_all_lattice_steps():
    all_steps = LatticeStep.all_step_vecs()
    assert len(all_steps) == 42
    assert len(set([s.tuple for s in all_steps])) == 42