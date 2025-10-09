import pytest
import numpy as np

from cnf.lattice.voronoi.math import SignedVoronoiValue, Sign, SignedValueSet, SignedVector, SignedVectorSet, ConormCalculator, Transformation, Conorm
from cnf.lattice.voronoi.voronoi_values import PrimaryVonorm, SecondaryVonorm, Conorm, VoronoiVector
from cnf.lattice.superbasis import get_v0_from_generating_vecs
from cnf.linalg import MatrixTuple, VectorTuple
from cnf.lattice.permutations import ConormPermutation
from cnf.linalg.unimodular import UNIMODULAR_MATRICES
from cnf.lattice.voronoi.conorm_list_form import ConormListForm

def test_can_init_voronoi_value():
    v = SignedVoronoiValue.negative_one(PrimaryVonorm(0))
    assert v.sign == Sign.NEGATIVE
    assert v.count == 1
    assert v.value == PrimaryVonorm(0)
    assert v.value != PrimaryVonorm(1)

def test_sign_multiplication():
    assert Sign.POSITIVE.multiply(Sign.NEGATIVE) == Sign.NEGATIVE
    assert Sign.NEGATIVE.multiply(Sign.POSITIVE) == Sign.NEGATIVE

    assert Sign.NEGATIVE.multiply(Sign.NEGATIVE) == Sign.POSITIVE
    assert Sign.POSITIVE.multiply(Sign.POSITIVE) == Sign.POSITIVE

def test_can_init_voronoi_value_from_ct():
    v = SignedVoronoiValue.from_signed_count(-2, PrimaryVonorm(0))
    assert v.count == 2
    assert v.sign == Sign.NEGATIVE

    v = SignedVoronoiValue.from_signed_count(5, PrimaryVonorm(0))
    assert v.count == 5
    assert v.sign == Sign.POSITIVE

def test_signed_value_equality():
    v1 = SignedVoronoiValue([Sign.NEGATIVE, 2, PrimaryVonorm(1)])
    
    v2 = SignedVoronoiValue([Sign.NEGATIVE, 2, PrimaryVonorm(1)])
    assert v1 == v2
    assert v2 == v1

    v3 = SignedVoronoiValue([Sign.NEGATIVE, 3, PrimaryVonorm(1)])
    assert v1 != v3
    assert v3 != v1

    v4 = SignedVoronoiValue([Sign.POSITIVE, 2, PrimaryVonorm(1)])
    assert v1 != v4
    assert v4 != v1
    v5 = SignedVoronoiValue([Sign.NEGATIVE, 2, PrimaryVonorm(2)])
    assert v1 != v5
    assert v5 != v1

def test_signed_value_sign_multiplication():
    pv1 = PrimaryVonorm(1)
    sv = SignedVoronoiValue([Sign.POSITIVE, 2, pv1])
    sv2 = sv.multiply_sign(Sign.NEGATIVE)
    assert sv2 == SignedVoronoiValue([Sign.NEGATIVE, 2, pv1])


def test_signed_value_set():
    svs = SignedValueSet([])
    pv0 = PrimaryVonorm(0)
    assert svs.get_count(pv0) == 0
    svs.add_val(SignedVoronoiValue.positive_one(pv0))
    assert svs.get_count(pv0) == 1
    svs.add_val(SignedVoronoiValue.positive_one(pv0))
    assert svs.get_count(pv0) == 2
    svs.add_val(SignedVoronoiValue.negative_one(pv0))
    assert svs.get_count(pv0) == 1
    svs.add_val(SignedVoronoiValue.negative_one(pv0))
    svs.add_val(SignedVoronoiValue.negative_one(pv0))
    assert svs.get_count(pv0) == -1

    pv1 = PrimaryVonorm(1)
    svs.add_val(SignedVoronoiValue.positive_one(pv1))
    assert svs.get_count(pv0) == -1
    assert svs.get_count(pv1) == 1

    c12 = Conorm((1, 2))
    svs.add_val(SignedVoronoiValue.positive_one(c12))
    assert svs.get_count(pv0) == -1
    assert svs.get_count(c12) == 1

    c21 = Conorm((2, 1))
    svs.add_val(SignedVoronoiValue.positive_one(c12))
    assert svs.get_count(pv0) == -1
    assert svs.get_count(c12) == 2
    assert svs.get_count(c21) == 2

def test_distribute_sign():
    v1 = SignedVoronoiValue.positive_one(PrimaryVonorm(1))
    v2 = SignedVoronoiValue.negative_one(SecondaryVonorm((0, 1)))
    svs = SignedValueSet([v1, v2])

    assert v1 in svs
    assert v2 in svs

    distributed = svs.distribute(Sign.NEGATIVE)
    assert v1.negative() in distributed
    assert v2.negative() in distributed

def test_signed_vector():
    v1 = SignedVector.negative_v1()
    v2 = SignedVector.positive_v2()
    result = v1.dot(v2)
    assert isinstance(result, SignedVoronoiValue)
    assert result.sign == Sign.NEGATIVE
    assert result.value == Conorm((1, 2))

    v1 = SignedVector.negative_v1()
    v2 = SignedVector.positive_v1()
    result = v1.dot(v2)
    assert isinstance(result, SignedVoronoiValue)
    assert result.sign == Sign.NEGATIVE
    assert result.value == PrimaryVonorm(1)

    v1 = SignedVector.positive_v1()
    v2 = SignedVector.positive_v1()
    result = v1.dot(v2)
    assert isinstance(result, SignedVoronoiValue)
    assert result.sign == Sign.POSITIVE
    assert result.value == PrimaryVonorm(1)

@pytest.fixture
def vector_set_1():
    return SignedVectorSet([
        SignedVector.negative_v1(),
        SignedVector.negative_v2()
    ])

@pytest.fixture
def vector_set_2():
    return SignedVectorSet([
        SignedVector.positive_v1(),
        SignedVector.positive_v3()
    ])

def test_signed_vector_set_multiplication(vector_set_1, vector_set_2):
    result = vector_set_1.multiply(vector_set_2)
    assert isinstance(result, SignedValueSet)
    assert SignedVoronoiValue.negative_one(PrimaryVonorm(1)) in result
    assert SignedVoronoiValue.negative_one(Conorm((1,3))) in result
    assert SignedVoronoiValue.negative_one(Conorm((1,2))) in result
    assert SignedVoronoiValue.negative_one(Conorm((2,3))) in result
    assert len(result) == 4

def test_vonorm_to_conorm_reduction(vector_set_1, vector_set_2):
    result = ConormCalculator.vonorms_to_conorms(vector_set_1.multiply(vector_set_2))
    assert isinstance(result, SignedValueSet)

    assert not SignedVoronoiValue.negative_one(PrimaryVonorm(1)) in result
    assert not SignedVoronoiValue.negative_one(Conorm((1,3))) in result
    assert not SignedVoronoiValue.negative_one(Conorm((1,2))) in result
    assert SignedVoronoiValue.negative_one(Conorm((2,3))) in result
    assert SignedVoronoiValue.positive_one(Conorm((0,1))) in result
    
    assert len(result) == 2

def test_transformation():
    mat = np.array([
        [1, 0, 0],
        [-1, -1, 1],
        [0, 0, -1]
    ])
    t = Transformation(MatrixTuple(mat))
    expected_v3 = get_v0_from_generating_vecs(mat.T)
    assert t.v3() == tuple(expected_v3)
    assert (t.v3().vector == -np.array([1, -1, -1])).all()

def test_col_to_vector_set():
    vector = [1, 0 ,1]
    vectors = ConormCalculator.col_to_vector_set(vector)
    assert SignedVector.positive_v0() in vectors
    assert SignedVector.negative_v0() not in vectors

    assert SignedVector.positive_v1() not in vectors
    assert SignedVector.negative_v0() not  in vectors

    assert SignedVector.positive_v2() in vectors
    assert SignedVector.negative_v2() not in vectors

    assert SignedVector.positive_v3() not in vectors
    assert SignedVector.negative_v3() not in vectors
    assert len(vectors) == 2

    vector = [0, -1 ,1]
    vectors = ConormCalculator.col_to_vector_set(vector)
    assert SignedVector.positive_v0() not in vectors
    assert SignedVector.negative_v0() not in vectors

    assert SignedVector.positive_v1() not in vectors
    assert SignedVector.negative_v1() in vectors

    assert SignedVector.positive_v2() in vectors
    assert SignedVector.negative_v2() not in vectors

    assert SignedVector.positive_v3() not in vectors
    assert SignedVector.negative_v3() not in vectors
    assert len(vectors) == 2

def test_remove_zero_conorms():
    c12 = SignedVoronoiValue.positive_one(Conorm((1, 2)))
    c13 = SignedVoronoiValue.negative_one(Conorm((1, 3)))
    c02 = SignedVoronoiValue.negative_one(Conorm((0, 2)))
    val_set = SignedValueSet([c12, c13, c02])
    zero_cs = set([Conorm((1,2)), Conorm((2,3)), Conorm((0, 2))])
    filtered = ConormCalculator.remove_zeros(val_set, zero_cs)
    assert Conorm((1,2)) not in filtered
    assert Conorm((1,3)) in filtered
    assert SignedVoronoiValue.negative_one(Conorm((1,3))) in filtered
    

def test_get_conorm():
    t = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, -1],
    ])
    t = Transformation(MatrixTuple(t))

    # Without Filtering
    cc = ConormCalculator(t)
    result = cc.get_conorm(Conorm((0, 1)), log=False)
    assert Conorm((0, 1)) in result
    assert Conorm((1, 2)) in result
    assert len(result) == 2

    result = cc.get_conorm(Conorm((0, 2)), log=False)
    assert Conorm((1, 2)) in result
    assert Conorm((2,3)) in result
    assert len(result) == 2

    result = cc.get_conorm(Conorm((0, 3)), log=False)
    assert Conorm((0, 3)) in result
    assert Conorm((1, 2)) in result
    assert len(result) == 2

    result = cc.get_conorm(Conorm((1, 2)), log=False)
    assert Conorm((1, 2)) in result
    assert len(result) == 1

    result = cc.get_conorm(Conorm((1, 3)), log=False)
    assert Conorm((1, 3)) in result
    assert Conorm((1, 2)) in result
    assert len(result) == 2

    result = cc.get_conorm(Conorm((2, 3)), log=False)
    assert Conorm((0, 2)) in result
    assert Conorm((1, 2)) in result
    assert len(result) == 2


    # With Filtering
    cc = ConormCalculator(t, set([Conorm((2, 3))]))
    result = cc.get_conorm(Conorm((0, 1)), log=False)
    assert Conorm((0, 1)) in result
    assert Conorm((1, 2)) in result
    assert len(result) == 2

    result = cc.get_conorm(Conorm((0, 2)), log=False)
    assert Conorm((1, 2)) in result
    assert Conorm((2,3)) not in result
    assert len(result) == 1

    result = cc.get_conorm(Conorm((0, 3)), log=False)
    assert Conorm((0, 3)) in result
    assert Conorm((1, 2)) in result
    assert len(result) == 2

    result = cc.get_conorm(Conorm((1, 2)), log=False)
    assert Conorm((1, 2)) in result
    assert len(result) == 1

    result = cc.get_conorm(Conorm((1, 3)), log=False)
    assert Conorm((1, 3)) in result
    assert Conorm((1, 2)) in result
    assert len(result) == 2

    result = cc.get_conorm(Conorm((2, 3)), log=False)
    assert Conorm((0, 2)) in result
    assert Conorm((1, 2)) in result
    assert len(result) == 2

    # print(cc.get_permutations())
    # assert ConormPermutation([0,1,4,3,2,5,6]) in cc.get_permutations()
    # assert ConormPermutation([0,1,4,3,2,6,5]) in cc.get_permutations()
    # assert len(cc.get_permutations()) == 2

def test_perm_should_be_present():
    mat = MatrixTuple.from_tuple((-1, 0, 0, -1, 0, 1, -1, 1, 0))
    mat = MatrixTuple.from_tuple((1, 0, 0, 0, 1, 0, 0, 0, 1))
    mat = MatrixTuple.from_tuple((0, 1, 0, 1, 0, 0, 0, 0, 1))
    print(mat.matrix)
    cf = ConormListForm([0])
    calc = ConormCalculator(Transformation(mat), cf.zero_conorms())
    print(cf.zero_conorms())
    perms = calc.get_permutations()
    print(perms)
    all_valid_perms = ConormPermutation.all_conorm_perm_tuples()
    assert len(set(perms).intersection(all_valid_perms)) > 0

def test_specific_perm_expectation_1():
    zero_indices = []
    cf = ConormListForm(zero_indices)
    perms_with_matrices = []
    mat_to_perms = {}
    for u in UNIMODULAR_MATRICES:
        t = Transformation(u)
        calc = ConormCalculator(t, cf.zero_conorms())
        try:
            mat_to_perms[u] = calc.get_permutations()
        except ValueError as e:
            print(e.__repr__())
    print(f"Considered zeros: {cf.zero_conorms()}")
    print(f"Found {len(mat_to_perms)} matrices total")

    def filter_invalid_perms(perm_list):
        return list(set(ConormPermutation.all_conorm_perm_tuples()).intersection(perm_list))
    
    filtered_mat_to_perms = { mat: filter_invalid_perms(plist) for mat, plist in mat_to_perms.items() }
    print(f"{len(filtered_mat_to_perms)} of them mapped to valid perms")
    all_perms = set([p for mat, perm_list in mat_to_perms.items() for p in perm_list])
    print(f"Found {len(all_perms)} perms")
    valid_found_perms = set(ConormPermutation.all_conorm_perm_tuples()).intersection(all_perms)
    print(f"Found {len(valid_found_perms)} valid perms")
    # print(all_perms)
    # for mat, perm in mat_to_perms.items():
    #     print(mat, perm)

def test_pathological_multiplication_case():

    vec1 = SignedVector.from_signed_count(-2, VoronoiVector.V2())
    vec2 = SignedVector.positive_one(VoronoiVector.V2())
    assert vec1.dot(vec2) == SignedVoronoiValue.from_signed_count(-2, PrimaryVonorm(2))

    mat = MatrixTuple.from_tuple((-1, 0, 0, -1, -1, 0, -1, -1, 1))
    t = Transformation(mat)
    print(mat.matrix)
    # [ -1 -1 -1]
    col1 = ConormCalculator.col_to_vector_set(t.get_col(0))
    assert col1 == SignedValueSet([SignedVector.positive_one(VoronoiVector(3))])
    # [0 0 1]
    col2 = ConormCalculator.col_to_vector_set(t.get_col(2))
    assert col2 == SignedValueSet([
        SignedVector.positive_one(VoronoiVector(2))
    ])
    # print(col1, col2)
    result = col1.multiply(col2)
    assert SignedVoronoiValue.from_signed_count(1, Conorm((2, 3))) in result
    
    assert len(result) == 1

def test_transformation_returns_int_cols_as_tuples():
    mat = MatrixTuple.from_tuple((-1, 0, 0, -1, -1, 0, -1, -1, 1))
    t = Transformation(mat)
    col = t.get_col(3)
    assert type(col) == VectorTuple
    for i in col:
        assert isinstance(i, int)
    assert col == (1, 2, 1)


def test_scale_voronoi_value():
    v = SignedVoronoiValue.negative_one(Conorm((1,2)))
    assert v.multiply_count(2) == SignedVoronoiValue.from_signed_count(-2, Conorm((1,2)))
    assert v.multiply_count(5) == SignedVoronoiValue.from_signed_count(-5, Conorm((1,2)))

    v = SignedVoronoiValue.positive_one(PrimaryVonorm(2))
    assert v.multiply_count(6) == SignedVoronoiValue.from_signed_count(6, PrimaryVonorm(2))
    assert v.multiply_count(5) == SignedVoronoiValue.from_signed_count(5, PrimaryVonorm(2))

def test_pathological_reduce():
    #SignedValueSet([(-, 1, P(1,2)), (-, 1, P(1,3)), (-, 2, (V_2)^2), (-, 3, P(2,3)), (-, 1, (V_3)^2)])
    svs = SignedValueSet([
        SignedVoronoiValue.negative_one(Conorm((1,2))),
        SignedVoronoiValue.negative_one(Conorm((1,3))),
        SignedVoronoiValue.from_signed_count(-2, PrimaryVonorm(2)),
        SignedVoronoiValue.from_signed_count(-3, Conorm((2,3))),
        SignedVoronoiValue.from_signed_count(-1, PrimaryVonorm(3)),
    ])
    result = ConormCalculator.vonorms_to_conorms(svs)

    expected = [
        SignedVoronoiValue.from_signed_count(2, Conorm((0,2))),
        SignedVoronoiValue.from_signed_count(1, Conorm((0,3))),
        SignedVoronoiValue.from_signed_count(1, Conorm((1,2))),
    ]

    for e in expected:
        assert e in result

    assert len(expected) == len(result)

        