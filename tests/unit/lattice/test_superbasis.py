import pytest
import numpy as np

from cnf.lattice import Superbasis
from cnf.lattice.permutations import CONORM_PERMUTATION_TO_VONORM_PERMUTATION, VONORM_PERMUTATION_TO_CONORM_PERMUTATION
from cnf.lattice.superbasis import get_v0_from_generating_vecs
from pymatgen.core.lattice import Lattice

def test_can_instantiate():
    test_lattice = Lattice.rhombohedral(1.5, 80)
    sb = Superbasis.from_pymatgen_lattice(test_lattice)

    assert (sb.superbasis_vecs[3] == -np.sum(test_lattice.matrix, axis=0)).all()
    assert (sb.superbasis_vecs[0] == test_lattice.matrix[0]).all()
    assert (sb.superbasis_vecs[1] == test_lattice.matrix[1]).all()
    assert (sb.superbasis_vecs[2] == test_lattice.matrix[2]).all()

    vonorms = sb.compute_vonorms()
    assert vonorms[3] == np.dot(-np.sum(test_lattice.matrix, axis=0), -np.sum(test_lattice.matrix, axis=0))
    assert vonorms[0] == np.dot(test_lattice.matrix[0], test_lattice.matrix[0])
    assert vonorms[1] == np.dot(test_lattice.matrix[1], test_lattice.matrix[1])
    assert vonorms[2] == np.dot(test_lattice.matrix[2], test_lattice.matrix[2])

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


def test_get_v0():
    lattice = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0,0,0.5]
    ])
    v0 = get_v0_from_generating_vecs(lattice)
    assert np.all(v0 == np.array([-2, -1, -0.5]))

def test_apply_permutation():
    vecs = np.array([
        [-1, -1, -1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    perm = (0, 1,)

def test_is_superbasis():
    hexarhombic_dodecahedron = Lattice([
        [-1, 0.2, 1],
        [1, -1, 0],
        [1, 1, 0],
    ])
    sb = Superbasis.from_pymatgen_lattice(hexarhombic_dodecahedron)
    assert sb.is_superbasis()

def test_voronoi_class_two_superbases_have_same_vonorms():
    hexarhombic_dodecahedron = Lattice([
        [-1, 0.2, 1],
        [1, -1, 0],
        [1, 1, 0],
    ])
    
    sb = Superbasis.from_pymatgen_lattice(hexarhombic_dodecahedron)
    # p_1_2 == 0 by design here - changes to choice of lattice vecs will break this
    assert sb.compute_vonorms().conorms[3] == 0
    assert np.all(sb.superbasis_vecs[3] == -(sb.superbasis_vecs[0] + sb.superbasis_vecs[1] + sb.superbasis_vecs[2]))
    assert sb.is_obtuse()
    old_vecs = sb.superbasis_vecs
    new_sb_vecs = np.array([
        old_vecs[0] + old_vecs[2],
        # old_vecs[1] + old_vecs[3],
        old_vecs[1],
        # old_vecs[2],
        -old_vecs[2],
        old_vecs[3] + old_vecs[2]
    ])
    assert np.all(new_sb_vecs[0] == np.array([0, 1.2, 1]))
    assert np.all(new_sb_vecs[1] == np.array([1, -1, 0]))
    assert np.all(new_sb_vecs[2] == np.array([-1, -1, 0]))
    assert np.all(new_sb_vecs[3] == np.array([0, 0.8, -1]))
    assert np.all(new_sb_vecs[3] == -(new_sb_vecs[0] + new_sb_vecs[1] + new_sb_vecs[2]))

    sb2 = Superbasis(new_sb_vecs)
    # print(sb.compute_vonorms().conorms)
    # print(sb2.compute_vonorms().conorms)
    # for cperm in sb.compute_vonorms().conorms.permissible_permutations:
    #     permuted_conorms = sb.compute_vonorms().conorms.apply_permutation(cperm)
    #     permuted_vonorms = sb.compute_vonorms().apply_permutation(cperm.to_vonorm_permutation())

    #     conorms_match = np.isclose(np.array(permuted_conorms.conorms), np.array(sb2.compute_vonorms().conorms.conorms)).all()
    #     vonorms_match = np.isclose(np.array(permuted_vonorms.vonorms), np.array(sb2.compute_vonorms().vonorms)).all()
    
    #     if conorms_match and vonorms_match:
    #         print(f"Matching Vonorm permutation: {cperm.to_vonorm_permutation()}")

    assert sb2.is_superbasis()
    assert sb.compute_vonorms().has_same_members(sb2.compute_vonorms())
    conorms = sb.compute_vonorms().conorms
    assert len(conorms.form.zero_indices) == 1
    assert conorms.form.voronoi_class == 2
    # print(f"{conorms.voronoi_class}: {len(conorms.permissible_permutations)}")    