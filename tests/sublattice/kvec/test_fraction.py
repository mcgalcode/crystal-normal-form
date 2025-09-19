import pytest

from cnf.sublattice.kvec import Fraction

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

def test_set_of_fractions():
    f1 = Fraction(1, 2)
    f2 = Fraction(1, 2)
    assert len(set([f1, f2])) == 1

def test_can_scale():
    f1 = Fraction(1, 3)
    assert f1.multiply(2) == Fraction(2, 3)

    f1 = Fraction(1, 3)
    assert f1.multiply(Fraction(3,4)) == Fraction(3, 12)


def test_convert_denominator():
    f1 = Fraction(1, 2)
    f2 = f1.convert_denominator(4)
    assert f2.num == 2
    assert f2.denom == 4

    f1 = Fraction(1, 2)
    f2 = f1.convert_denominator(2)
    assert f2.num == 1
    assert f2.denom == 2

    f1 = Fraction(2, 3)
    f2 = f1.convert_denominator(12)
    assert f2.num == 8
    assert f2.denom == 12

def test_common_denominator():
    f1 = Fraction(2, 3)
    f2 = Fraction(3,5)
    assert f1.common_denominator(f2) == 15
    assert f2.common_denominator(f1) == 15

    f1 = Fraction(2, 3)
    f2 = Fraction(3,3)
    assert f1.common_denominator(f2) == 3
    assert f2.common_denominator(f1) == 3

def test_add():
    f1 = Fraction(1, 2)
    f2 = Fraction(1, 4)
    assert f1.add(f2) == f2.add(f1)
    assert f1.add(f2) == Fraction(3, 4)

    f1 = Fraction(1, 1)
    f2 = Fraction(3, 7)
    assert f1.add(f2) == f2.add(f1)
    assert f1.add(f2) == Fraction(10, 7)

    f1 = Fraction(1, 3)
    f2 = Fraction(1, 3)
    assert f1.add(f2) == f2.add(f1)
    assert f1.add(f2) == Fraction(2, 3)