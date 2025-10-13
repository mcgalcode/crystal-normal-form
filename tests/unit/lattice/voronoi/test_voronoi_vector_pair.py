import pytest

from cnf.lattice.voronoi.vector_pair import VoronoiVectorPair

def test_can_instantiate_voronoi_vector_pair():
    pair = VoronoiVectorPair(0, 1)
    assert pair == (0, 1)

def test_validates_length():
    with pytest.raises(ValueError) as e:
        pair = VoronoiVectorPair(0,1,2)
    
    assert "exactly two" in str(e.value)

def test_validates_value():
    with pytest.raises(ValueError) as e:
        pair = VoronoiVectorPair(9,1)
    
    assert "non-viable" in str(e.value)

def test_sorts_values():
    assert VoronoiVectorPair(0, 1) == VoronoiVectorPair(1, 0)