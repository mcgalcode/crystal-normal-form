import pytest

from cnf.lattice.selling import SellingPair

def test_can_instantiate_selling_pair():
    pair = SellingPair(0, 1)
    assert pair == (0, 1)

def test_validates_length():
    with pytest.raises(ValueError) as e:
        pair = SellingPair(0,1,2)
    
    assert "exactly two" in str(e.value)

def test_validates_value():
    with pytest.raises(ValueError) as e:
        pair = SellingPair(9,1)
    
    assert "non-viable" in str(e.value)

def test_sorts_values():
    assert SellingPair(0, 1) == SellingPair(1, 0)