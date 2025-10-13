import pytest

from cnf.lattice.voronoi.voronoi_values import PrimaryVonorm, SecondaryVonorm, Conorm, VoronoiValue, VoronoiVector
from cnf.lattice.voronoi.vector_pair import VoronoiVectorPair

def test_can_instantiate_conorm():
    c = Conorm((1, 2))
    with pytest.raises(ValueError):
        Conorm((1, 1))
    assert Conorm((1, 2)) == Conorm((2, 1))

def test_can_hash_conorms():
    my_set = set()
    my_set.add(Conorm((1,2)))
    my_set.add(Conorm((1,2)))
    assert len(my_set) == 1
    my_set.add(Conorm((2, 1)))
    assert len(my_set) == 1
    my_set.add(Conorm((2, 3)))
    assert len(my_set) == 2


def test_can_instantiate_primary_vonorm():
    v = PrimaryVonorm(1)
    assert v == PrimaryVonorm(1)
    assert PrimaryVonorm(2) != PrimaryVonorm(1)
    with pytest.raises(ValueError):
        PrimaryVonorm(5)
    
def test_can_hash_primary_vonorms():
    my_set = set()
    my_set.add(PrimaryVonorm(2))
    my_set.add(PrimaryVonorm(2))
    assert len(my_set) == 1
    my_set.add(PrimaryVonorm(1))
    assert len(my_set) == 2
    my_set.add(PrimaryVonorm(3))
    assert len(my_set) == 3
    my_set.add(PrimaryVonorm(3))
    assert len(my_set) == 3

def test_can_instantiate_secondary_vonorm():
    v = SecondaryVonorm((0, 1))
    assert v.is_canonical
    assert v.complement == SecondaryVonorm((3, 2))
    assert not v.complement.is_canonical

    with pytest.raises(ValueError):
        SecondaryVonorm((2,2))

    with pytest.raises(ValueError):
        SecondaryVonorm((2,4))

def test_can_hash_secondary_vonorms():
    my_set = set()
    my_set.add(SecondaryVonorm((1,2)))
    my_set.add(SecondaryVonorm((1,2)))
    assert len(my_set) == 1
    my_set.add(SecondaryVonorm((2, 1)))
    assert len(my_set) == 1
    my_set.add(SecondaryVonorm((2, 3)))
    assert len(my_set) == 2

def test_can_dot_voronoi_vectors():
    # Conorms
    v1 = VoronoiVector(1)
    v2 = VoronoiVector(2)
    assert v1.dot(v2) == Conorm((1, 2))
    assert v2.dot(v1) == Conorm((1, 2))
    assert v2.dot(v1) != Conorm((3, 2))

    assert v1.dot(v1) == PrimaryVonorm(1)
    assert v1.dot(v1) != PrimaryVonorm(2)