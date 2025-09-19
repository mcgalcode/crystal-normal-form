import pytest

from cnf.sublattice.kvec.utils import get_divisors, valid_denominator_sets

def test_divisors():
    assert set(get_divisors(4)) == {1, 2, 4}
    assert set(get_divisors(12)) == {1, 2, 3, 4, 6, 12}

def test_denominator_sets():
    print(valid_denominator_sets(4))





