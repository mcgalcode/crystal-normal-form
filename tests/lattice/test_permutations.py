import pytest
import numpy as np

from cnf.lattice.permutations import VonormPermutation, ConormPermutation, VONORM_PERMUTATION_TO_CONORM_PERMUTATION, compose_permutations, apply_permutation


def test_vonorm_permutation():
    permutation = VonormPermutation((6, 5, 2, 3, 4, 1, 0))
    cpermutation = permutation.to_conorm_permutation()
    assert cpermutation == (0, 1, 4, 3, 2, 6, 5)

def test_equality():
    assert VonormPermutation((1,2,5,3,4,6,0)) == VonormPermutation((1,2,5,3,4,6,0))
    assert ConormPermutation((1,2,5,3,4,6,0)) == ConormPermutation((1,2,5,3,4,6,0))

def test_item_access():
    permutation = VonormPermutation((6, 5, 2, 3, 4, 1, 0))
    assert permutation[0] == 6
    assert permutation[-1] == 0

def test_unimodularity():
    for vperm in VONORM_PERMUTATION_TO_CONORM_PERMUTATION:
        u = VonormPermutation(vperm).to_unimodular_matrix()
        assert np.abs(np.linalg.det(u)) == 1

def test_apply_permutation():
    perm = (3,1,2,0)
    vals = ['a', 'b', 'c', 'd']
    result = apply_permutation(vals, perm)
    assert tuple(result) == tuple(['d', 'b', 'c', 'a'])

def test_compose_permutations():
    perm1 = (0, 2, 1)
    perm2 = (1, 0, 2)

    composition_1 = compose_permutations(perm1, perm2)
    assert composition_1 == (1, 2, 0)

    composition_2 = compose_permutations(perm2, perm1)
    assert composition_2 == (2, 0, 1)

def test_are_vonorm_permutations_an_s7_subgroup():
    vperms = set(VONORM_PERMUTATION_TO_CONORM_PERMUTATION.keys())
    
    for p1 in vperms:
        for p2 in vperms:
            composed = compose_permutations(p1, p2)
            assert composed in vperms


def test_are_conorm_permutations_an_s7_subgroup():
    vperms = set(VONORM_PERMUTATION_TO_CONORM_PERMUTATION.values())
    
    for p1 in vperms:
        for p2 in vperms:
            composed = compose_permutations(p1, p2)
            assert composed in vperms
