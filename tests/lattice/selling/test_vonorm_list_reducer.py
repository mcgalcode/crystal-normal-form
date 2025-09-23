import pytest

from cnf.lattice.selling import VonormListSellingReducer, SuperbasisSellingReducer
from cnf.lattice import Superbasis
from cnf.lattice.rounding import DiscretizedVonormComputer


def test_can_selling_reduce_vonorm_list(monoclinic_lattice):
    vl = Superbasis.from_pymatgen_lattice(monoclinic_lattice).compute_vonorms()
    assert not vl.is_obtuse()
    tol = 1e-5
    reducer = VonormListSellingReducer(tol)
    reduce_result = reducer.reduce(vl)
    assert reduce_result.reduced_object.is_obtuse(tol=tol)

def test_can_selling_reduce_acute_lattice_vonorm_list(acute_lattice_1):
    basis = Superbasis.from_pymatgen_lattice(acute_lattice_1)
    assert not basis.is_obtuse()
    vonorm_list = basis.compute_vonorms()
    assert vonorm_list.is_superbasis()

    tol = 1e-5
    reducer = VonormListSellingReducer(tol, verbose_logging=True)
    reduce_result = reducer.reduce(vonorm_list)

    assert reduce_result.num_steps > 0
    assert reduce_result.reduced_object.is_obtuse(tol=1e-5)

@pytest.mark.skip
def test_can_reduce_discretized_very_acute_vonorm_list(acute_lattice_1):
    discretization = 1.0
    dvc = DiscretizedVonormComputer(discretization)
    basis = Superbasis.from_pymatgen_lattice(acute_lattice_1)
    assert not basis.is_obtuse()
    vonorm_list = basis.compute_vonorms()

    discretized = dvc.find_closest_valid_vonorms(vonorm_list)
    # print(discretized.primary_sum(), discretized.secondary_sum())
    assert discretized.is_superbasis()

    tol = 1e-8
    sb = discretized.to_superbasis(discretization)
    sb_reducer = SuperbasisSellingReducer(tol, verbose_logging=True, max_steps=20)
    sb_reduce_result = sb_reducer.reduce(sb)
    assert sb_reduce_result.num_steps > 0
    assert sb_reduce_result.reduced_object.is_obtuse()   

    reducer = VonormListSellingReducer(tol, verbose_logging=True, max_steps=20)
    reduce_result = reducer.reduce(discretized)

    assert reduce_result.num_steps > 0
    assert reduce_result.reduced_object.is_obtuse()   

def test_can_reduce_with_small_discretization_barely_obtuse(barely_obtuse_lattice):
    VERY_SMALL_DISCRETIZATION = 0.1
    basis = Superbasis.from_pymatgen_lattice(barely_obtuse_lattice)
    assert not basis.is_obtuse()
    vonorm_list = basis.compute_vonorms()

    dvc = DiscretizedVonormComputer(VERY_SMALL_DISCRETIZATION)
    discretized = dvc.find_closest_valid_vonorms(vonorm_list)
    assert discretized.is_superbasis()

    tol = 1e-5
    reducer = VonormListSellingReducer(tol, verbose_logging=True)
    reduce_result = reducer.reduce(discretized)

    assert reduce_result.num_steps > 0
    assert reduce_result.reduced_object.is_obtuse()    

def test_can_selling_reduce_discretized_vonorm_list(acute_lattice_1):
    basis = Superbasis.from_pymatgen_lattice(acute_lattice_1)
    assert not basis.is_obtuse()
    vonorm_list = basis.compute_vonorms()
    dvc = DiscretizedVonormComputer(1.2)
    discretized = dvc.find_closest_valid_vonorms(vonorm_list)
    assert discretized.is_superbasis()

    tol = 1e-5
    reducer = VonormListSellingReducer(tol, verbose_logging=False)
    reduce_result = reducer.reduce(discretized)

    assert reduce_result.num_steps > 0
    assert reduce_result.reduced_object.is_obtuse()