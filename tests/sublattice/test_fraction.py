import pytest

from cnf.sublattice import Fraction

def test_equality():
    f1 = Fraction(1, 2)
    f2 = Fraction(1, 2)
    assert f1 == f2

    f1 = Fraction(1, 2)
    f2 = Fraction(2, 4)
    assert f1 == f2

def test_simplify():
    f = Fraction(12, 6)
    s = f.simplify()
    assert s.num == 2 and s.denom == 1

def test_mod_one():
    f = Fraction(12, 6)
    m = f.mod_one()
    assert m.num == 0, m.denom == 6

def test_is_multiple():
    f1 = Fraction(2, 3)
    f2 = Fraction(1, 3)
    assert f1.is_multiple_of(f2) == 2

    f1 = Fraction(1, 3)
    f2 = Fraction(4, 6)
    assert f2.is_multiple_of(f1) == 2

    f1 = Fraction(2, 6)
    f2 = Fraction(4, 6)
    assert f2.is_multiple_of(f1) == 2

    f1 = Fraction(2, 6)
    f2 = Fraction(3, 6)
    assert not f2.is_multiple_of(f1)

    f1 = Fraction(0, 6)
    f2 = Fraction(3, 6)
    assert not f2.is_multiple_of(f1)
    assert not f1.is_multiple_of(f2)