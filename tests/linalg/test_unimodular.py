import pytest

from cnf.linalg.unimodular import UNIMODULAR_MATRICES


def test_load_unimodular():
    for m in UNIMODULAR_MATRICES:
        assert m.determinant() == 1
        assert m.is_unimodular()