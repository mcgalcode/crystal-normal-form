import pytest
import numpy as np

from cnf.lattice.permutations import VonormPermutation, ConormPermutation, VONORM_PERMUTATION_TO_CONORM_PERMUTATION


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