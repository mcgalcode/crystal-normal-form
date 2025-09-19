import pytest

from cnf.lattice.selling import VonormListSellingReducer
from cnf.lattice import Superbasis, VonormList
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

@pytest.mark.xfail
def test_can_reduce_discretized_very_acute_vonorm_list(acute_lattice_1):
    discretization = 1.5
    basis = Superbasis.from_pymatgen_lattice(acute_lattice_1)
    assert not basis.is_obtuse()
    vonorm_list = basis.compute_vonorms()

    discretized = DiscretizedVonormComputer(vonorm_list, lattice_step_size=discretization).find_closest_valid_vonorms()
    discretized_list = VonormList(discretized)
    assert discretized_list.is_superbasis()

    tol = 1e-5
    reducer = VonormListSellingReducer(tol, verbose_logging=True, max_steps=20)
    reduce_result = reducer.reduce(discretized_list)

    assert reduce_result.num_steps > 0
    assert reduce_result.reduced_object.is_obtuse()   

def test_can_reduce_with_small_discretization_barely_obtuse(barely_obtuse_lattice):
    VERY_SMALL_DISCRETIZATION = 0.1
    basis = Superbasis.from_pymatgen_lattice(barely_obtuse_lattice)
    assert not basis.is_obtuse()
    vonorm_list = basis.compute_vonorms()

    discretized = DiscretizedVonormComputer(vonorm_list, lattice_step_size=VERY_SMALL_DISCRETIZATION).find_closest_valid_vonorms()
    discretized_list = VonormList(discretized)
    assert discretized_list.is_superbasis()

    tol = 1e-5
    reducer = VonormListSellingReducer(tol, verbose_logging=True)
    reduce_result = reducer.reduce(discretized_list)

    assert reduce_result.num_steps > 0
    assert reduce_result.reduced_object.is_obtuse()    

def test_can_selling_reduce_discretized_vonorm_list(acute_lattice_1):
    basis = Superbasis.from_pymatgen_lattice(acute_lattice_1)
    assert not basis.is_obtuse()
    vonorm_list = basis.compute_vonorms()

    discretized = DiscretizedVonormComputer(vonorm_list, lattice_step_size=1.2).find_closest_valid_vonorms()
    discretized_list = VonormList(discretized)
    assert discretized_list.is_superbasis()

    tol = 1e-5
    reducer = VonormListSellingReducer(tol, verbose_logging=False)
    reduce_result = reducer.reduce(discretized_list)

    assert reduce_result.num_steps > 0
    assert reduce_result.reduced_object.is_obtuse()