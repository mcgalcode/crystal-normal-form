import pytest
import numpy as np

from cnf.lattice import Superbasis, VonormList
from cnf.lattice.lattice_normal_form import LatticeNormalForm
from pymatgen.core.lattice import Lattice


def test_can_get_conorms():
    test_lattice = Lattice.orthorhombic(1.1, 1.5, 2.0)
    basis = Superbasis.from_pymatgen_lattice(test_lattice)
    vlist = basis.compute_vonorms()

    conorms = vlist.conorms

    assert np.isclose(np.dot(basis.superbasis_vecs[0], basis.superbasis_vecs[1]), conorms[0])
    assert np.isclose(np.dot(basis.superbasis_vecs[0], basis.superbasis_vecs[2]), conorms[1])
    assert np.isclose(np.dot(basis.superbasis_vecs[0], basis.superbasis_vecs[3]), conorms[2])
    assert np.isclose(np.dot(basis.superbasis_vecs[1], basis.superbasis_vecs[2]), conorms[3])
    assert np.isclose(np.dot(basis.superbasis_vecs[1], basis.superbasis_vecs[3]), conorms[4])
    assert np.isclose(np.dot(basis.superbasis_vecs[2], basis.superbasis_vecs[3]), conorms[5])

def test_can_check_obtuseness():
    nonobtuse_lattice = Lattice.rhombohedral(1.5,75)
    basis = Superbasis.from_pymatgen_lattice(nonobtuse_lattice)
    assert not basis.compute_vonorms().is_obtuse()

    obtuse_lattice = Lattice.orthorhombic(1.5,2, 2.5)
    basis = Superbasis.from_pymatgen_lattice(obtuse_lattice)
    assert basis.compute_vonorms().is_obtuse()

def test_can_hash_for_use_in_set():
    vlist1 = VonormList([1,2,3,4,5,6,7])
    vlist2 = VonormList([1,2,3,4,5,6,7])
    vlist3 = VonormList([1,2,3,4,5,6,8])

    vlist_set = {vlist1, vlist2}
    assert len(vlist_set) == 1

    vlist_set.add(vlist3)
    assert len(vlist_set) == 2
