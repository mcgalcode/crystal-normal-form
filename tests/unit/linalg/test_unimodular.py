import pytest

from cnf.linalg.unimodular import UNIMODULAR_MATRICES

def test_load_unimodular():
    assert UNIMODULAR_MATRICES is not None