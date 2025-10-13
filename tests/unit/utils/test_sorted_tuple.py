import pytest

from cnf.utils.sorted_tuple import SortedTuple

def test_can_instantiate_sorted_tuple():
    pair = SortedTuple(0, 1)
    assert pair == (0, 1)

def test_sorts_values():
    assert SortedTuple(0, 1, 3, 4) == SortedTuple(1, 4, 0, 3)