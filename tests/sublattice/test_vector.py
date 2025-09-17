import pytest

from cnf.sublattice import Vector, Fraction


def test_is_multiple():
    v1 = Vector([Fraction.zero(), Fraction.zero(), Fraction(1, 4)])
    v2 = Vector([Fraction.zero(), Fraction.zero(), Fraction(3, 4)])
    assert v2.is_multiple_of(v1)

    v1 = Vector([Fraction.zero(), Fraction(1,4), Fraction(1, 2)])
    v2 = Vector([Fraction.zero(), Fraction(1,4), Fraction(1, 4)])
    assert not v2.is_multiple_of(v1)
    assert not v1.is_multiple_of(v2)

def test_can_scale_self():
    v1 = Vector([Fraction(1, 4), Fraction(2, 4), Fraction(2, 3)])

    s1 = Fraction.whole_number(2)
    assert v1.scale(s1) == Vector([Fraction(2, 4), Fraction(4, 4), Fraction(4, 3)])

    v = Vector([Fraction(1, 4), Fraction(2, 4), Fraction(2, 3)])
    s = Fraction(2, 5)
    assert v.scale(s) == Vector([Fraction(2, 20), Fraction(4, 20), Fraction(4, 15)])

def test_can_mod_one():
    v = Vector([Fraction(5, 4), Fraction(7, 4), Fraction(10, 3)])
    assert v.mod_one() == Vector([Fraction(1, 4), Fraction(3, 4), Fraction(1, 3)])

    v = Vector([Fraction(1, 3), Fraction(7, 4), Fraction(10, 3)])
    assert v.mod_one() == Vector([Fraction(1, 3), Fraction(3, 4), Fraction(1, 3)])

def test_share_cyclic_group():
    v1 = Vector([Fraction(1,4), Fraction.zero(), Fraction(3, 4)])
    v2 = Vector([Fraction(3,4), Fraction.zero(), Fraction(1, 4)])
    assert v1.in_same_cyclic_group(v2, 4)

# Excluding: [0, 1/4, 1/2] because [0, 1/4, 1/4] is already there
# Excluding: [0, 1/4, 3/4] because [0, 1/4, 1/4] is already there
# Excluding: [0, 1/2, 1/4] because [0, 1/4, 1/4] is already there
# Excluding: [0, 3/4, 0] because [0, 1/4, 0] is already there
# Excluding: [0, 3/4, 1/4] because [0, 1/4, 1/4] is already there
# Excluding: [0, 3/4, 3/4] because [0, 1/4, 1/4] is already there
# Excluding: [1/4, 0, 1/2] because [1/4, 0, 1/4] is already there
# Excluding: [1/4, 0, 3/4] because [1/4, 0, 1/4] is already there