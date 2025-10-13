import pytest

from cnf.sublattice.kvec import CyclicGroup, FractionVector, Fraction

def test_cyclic_group_equality():
    cg1 = CyclicGroup.from_generator(FractionVector([Fraction(1,4), Fraction(1, 2), Fraction(3, 4)]), 4)
    cg2 = CyclicGroup.from_generator(FractionVector([Fraction(3,4), Fraction(1, 2), Fraction(1, 4)]), 4)
    assert cg1 == cg2

def test_cyclic_group_hashing():
    cg1 = CyclicGroup.from_generator(FractionVector([Fraction(1,4), Fraction(1, 2), Fraction(3, 4)]), 4)
    cg2 = CyclicGroup.from_generator(FractionVector([Fraction(3,4), Fraction(1, 2), Fraction(1, 4)]), 4)
    cgs = set([cg1, cg2])
    assert len(cgs) == 1