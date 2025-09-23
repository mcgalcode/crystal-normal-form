import pytest
import numpy as np

from cnf.lattice.selling.selling_reduction_result import SellingReductionResult
from cnf.lattice.selling import SellingTransformMatrix
from cnf.lattice.selling import VoronoiVectorPair

def test_selling_result_collapses_matrices():
    mat1 = SellingTransformMatrix.from_pair(VoronoiVectorPair(0, 1))
    mat2 = SellingTransformMatrix.from_pair(VoronoiVectorPair(2, 3))
    mat3 = SellingTransformMatrix.from_pair(VoronoiVectorPair(1, 3))

    res = SellingReductionResult(None, [mat1, mat2, mat3], 1, 0)
    assert res.transform_matrix == mat1 @ mat2 @ mat3