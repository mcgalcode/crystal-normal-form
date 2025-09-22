import pytest
import numpy as np

from cnf.lattice import Superbasis, VonormList
from cnf.lattice.lattice_normal_form import LatticeNormalForm
from pymatgen.core.lattice import Lattice
from cnf.lattice.rounding import DiscretizedVonormComputer


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

def test_can_round_trip_to_superbasis():
    l = Lattice.cubic(1.2)
    sb = Superbasis.from_pymatgen_lattice(l)
    vonorms = sb.compute_vonorms()
    recovered_sb = vonorms.to_superbasis()
    assert recovered_sb.compute_vonorms().has_same_members(sb.compute_vonorms())

    l = Lattice.rhombohedral(1.2, 65)
    sb = Superbasis.from_pymatgen_lattice(l)
    vonorms = sb.compute_vonorms()
    recovered_sb = vonorms.to_superbasis()
    assert recovered_sb.compute_vonorms().has_same_members(sb.compute_vonorms())

    l = Lattice.from_parameters(1, 2, 3, 85, 80, 88)
    sb = Superbasis.from_pymatgen_lattice(l)
    vonorms = sb.compute_vonorms()
    recovered_sb = vonorms.to_superbasis()
    assert recovered_sb.compute_vonorms().has_same_members(sb.compute_vonorms())

    l = Lattice.from_parameters(1, 2, 3, 25, 35, 60)
    sb = Superbasis.from_pymatgen_lattice(l)
    vonorms = sb.compute_vonorms()
    recovered_sb = vonorms.to_superbasis()
    assert recovered_sb.compute_vonorms().has_same_members(sb.compute_vonorms())

    l = Lattice.from_parameters(1.5, 1, 2, 30, 50, 130)
    sb = Superbasis.from_pymatgen_lattice(l)
    vonorms = sb.compute_vonorms()
    recovered_sb = vonorms.to_superbasis()
    assert recovered_sb.compute_vonorms().has_same_members(sb.compute_vonorms())

def test_can_recover_superbasis_after_discretization():
    xi = 0.01
    dvc = DiscretizedVonormComputer(xi)

    l = Lattice.cubic(1.2)
    sb = Superbasis.from_pymatgen_lattice(l)
    vonorms = dvc.find_closest_valid_vonorms(sb.compute_vonorms())
    print(sb.compute_vonorms())
    print(vonorms)
    gemini_gen = vonorms.recover_generators(xi)

    assert Superbasis.from_generating_vecs(gemini_gen).is_superbasis()
    assert Superbasis.from_generating_vecs(gemini_gen).compute_vonorms().has_same_members(sb.compute_vonorms())

def test_can_round_trip_to_superbasis_after_discretization():
    xi = 0.01
    dvc = DiscretizedVonormComputer(xi)

    l = Lattice.cubic(1.2)
    sb = Superbasis.from_pymatgen_lattice(l)
    vonorms = dvc.find_closest_valid_vonorms(sb.compute_vonorms())
    recovered_sb = vonorms.to_superbasis(xi)
    assert recovered_sb.compute_vonorms().has_same_members(sb.compute_vonorms(), 0.02)

    l = Lattice.rhombohedral(1.2, 65)
    sb = Superbasis.from_pymatgen_lattice(l)
    vonorms = dvc.find_closest_valid_vonorms(sb.compute_vonorms())
    recovered_sb = vonorms.to_superbasis(xi)
    assert recovered_sb.compute_vonorms().has_same_members(sb.compute_vonorms(), 0.02)

    l = Lattice.from_parameters(1, 2, 3, 85, 80, 88)
    sb = Superbasis.from_pymatgen_lattice(l)
    vonorms = dvc.find_closest_valid_vonorms(sb.compute_vonorms())
    recovered_sb = vonorms.to_superbasis(xi)
    assert recovered_sb.compute_vonorms().has_same_members(sb.compute_vonorms(), 0.02)

    l = Lattice.from_parameters(1, 2, 3, 25, 35, 60)
    sb = Superbasis.from_pymatgen_lattice(l)
    vonorms = dvc.find_closest_valid_vonorms(sb.compute_vonorms())
    recovered_sb = vonorms.to_superbasis(xi)
    assert recovered_sb.compute_vonorms().has_same_members(sb.compute_vonorms(), 0.02)

    l = Lattice.from_parameters(1.5, 1, 2, 30, 50, 130)
    sb = Superbasis.from_pymatgen_lattice(l)
    vonorms = dvc.find_closest_valid_vonorms(sb.compute_vonorms())
    recovered_sb = vonorms.to_superbasis(xi)
    assert recovered_sb.compute_vonorms().has_same_members(sb.compute_vonorms(), 0.02)
