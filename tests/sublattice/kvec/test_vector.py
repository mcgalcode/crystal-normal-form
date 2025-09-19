import pytest

from cnf.sublattice.kvec import FractionVector, Fraction


def test_is_multiple():
    v1 = FractionVector([Fraction.zero(), Fraction.zero(), Fraction(1, 4)])
    v2 = FractionVector([Fraction.zero(), Fraction.zero(), Fraction(3, 4)])
    assert v2.is_multiple_of(v1)

    v1 = FractionVector([Fraction.zero(), Fraction(1,4), Fraction(1, 2)])
    v2 = FractionVector([Fraction.zero(), Fraction(1,4), Fraction(1, 4)])
    assert not v2.is_multiple_of(v1)
    assert not v1.is_multiple_of(v2)

def test_can_scale_self():
    v1 = FractionVector([Fraction(1, 4), Fraction(2, 4), Fraction(2, 3)])

    s1 = Fraction.whole_number(2)
    assert v1.scale(s1) == FractionVector([Fraction(2, 4), Fraction(4, 4), Fraction(4, 3)])

    v = FractionVector([Fraction(1, 4), Fraction(2, 4), Fraction(2, 3)])
    s = Fraction(2, 5)
    assert v.scale(s) == FractionVector([Fraction(2, 20), Fraction(4, 20), Fraction(4, 15)])

def test_can_mod_one():
    v = FractionVector([Fraction(5, 4), Fraction(7, 4), Fraction(10, 3)])
    assert v.mod_one() == FractionVector([Fraction(1, 4), Fraction(3, 4), Fraction(1, 3)])

    v = FractionVector([Fraction(1, 3), Fraction(7, 4), Fraction(10, 3)])
    assert v.mod_one() == FractionVector([Fraction(1, 3), Fraction(3, 4), Fraction(1, 3)])

def test_share_cyclic_group():
    v1 = FractionVector([Fraction(1,4), Fraction.zero(), Fraction(3, 4)])
    v2 = FractionVector([Fraction(3,4), Fraction.zero(), Fraction(1, 4)])
    assert v1.in_same_cyclic_group(v2, 4)
    assert v2.in_same_cyclic_group(v1, 4)

def test_set_of_vectors():
    f1 = Fraction(1, 2)
    f2 = Fraction(1, 3)
    f3 = Fraction(1, 4)
    vec = FractionVector([f1, f2, f3])
    vec2 = FractionVector([f1, f2, f3])
    assert vec == vec2

    assert len(set([vec, vec2])) == 1