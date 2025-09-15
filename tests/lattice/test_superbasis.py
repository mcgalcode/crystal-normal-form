import pytest
import numpy as np

from cnf.lattice import Superbasis
from pymatgen.core.lattice import Lattice

def test_can_instantiate():
    test_lattice = Lattice.rhombohedral(1.5, 80)
    sb = Superbasis.from_pymatgen_lattice(test_lattice)

    assert (sb.superbasis_vecs[0] == -np.sum(test_lattice.matrix, axis=0)).all()
    assert (sb.superbasis_vecs[1] == test_lattice.matrix[0]).all()
    assert (sb.superbasis_vecs[2] == test_lattice.matrix[1]).all()
    assert (sb.superbasis_vecs[3] == test_lattice.matrix[2]).all()

    vonorms = sb.compute_vonorms()
    assert vonorms[0] == np.dot(-np.sum(test_lattice.matrix, axis=0), -np.sum(test_lattice.matrix, axis=0))
    assert vonorms[1] == np.dot(test_lattice.matrix[0], test_lattice.matrix[0])
    assert vonorms[2] == np.dot(test_lattice.matrix[1], test_lattice.matrix[1])
    assert vonorms[3] == np.dot(test_lattice.matrix[2], test_lattice.matrix[2])

    assert vonorms[4] == np.dot(-test_lattice.matrix[1] - test_lattice.matrix[2], -test_lattice.matrix[1] - test_lattice.matrix[2])
    assert vonorms[5] == np.dot(-test_lattice.matrix[0] - test_lattice.matrix[2], -test_lattice.matrix[0] - test_lattice.matrix[2])
    assert vonorms[6] == np.dot(-test_lattice.matrix[0] - test_lattice.matrix[1], -test_lattice.matrix[0] - test_lattice.matrix[1])

def test_can_transform():

    generating_vecs = np.array([
        [0, 0, 2],
        [0, 3, 0],
        [2, 1, 0]
    ])

    basis = Superbasis.from_generating_vecs(generating_vecs)

    transform = np.array([
        [0, 0, 0],
        [0, 2, 0],
        [1, 0, 3]
    ])
    new_sb = basis.apply_matrix_transform(transform)
    
    new_vecs = new_sb.generating_vecs()
    assert (new_vecs[0] == [2, 1, 0]).all()
    assert (new_vecs[1] == [0, 6, 0]).all()
    assert (new_vecs[2] == [6, 3, 0]).all()