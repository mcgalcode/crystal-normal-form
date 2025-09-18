import pytest
import numpy as np

from cnf.lattice import VonormList, Superbasis
from cnf.lattice.utils import selling_reduce
from cnf.lattice.rounding import DiscretizedVonormComputer
from pymatgen.core.lattice import Lattice

@pytest.fixture
def monoclinic_lattice():
    return Lattice.from_parameters(1.5, 1, 2, 30, 50, 130)

@pytest.fixture
def rhombohedral_lattice():
    return Lattice.rhombohedral(1.2, 65)

def test_can_selling_transform(monoclinic_lattice):
    basis = Superbasis.from_pymatgen_lattice(monoclinic_lattice)
    vonorm_list = basis.compute_vonorms()

    transformed_basis, swap = basis.selling_transform()
    print(transformed_basis.compute_vonorms())

    transformed_vonorms, _ = vonorm_list.selling_transform()
    print("VIA VONORM")
    print(vonorm_list)
    print(transformed_vonorms)

    assert np.isclose(transformed_basis.compute_vonorms().vonorms, transformed_vonorms.vonorms).all()

def test_can_selling_reduce_discretized_vonorm_list(monoclinic_lattice):
    basis = Superbasis.from_pymatgen_lattice(monoclinic_lattice)
    vonorm_list = basis.compute_vonorms()

    discretized = DiscretizedVonormComputer(vonorm_list.vonorms, lattice_step_size=0.05).find_closest_valid_vonorms()
    discretized_list = VonormList(discretized)

    reduced, num_steps = selling_reduce(discretized_list)
    print(num_steps)
    assert num_steps > 0
    assert reduced.is_obtuse()

def test_can_selling_reduce_superbasis(monoclinic_lattice):
    sb = Superbasis.from_pymatgen_lattice(monoclinic_lattice)
    tol = 1e-7
    reduced, _ = selling_reduce(sb, tol=tol)
    assert reduced.is_obtuse(tol=tol)

def test_selling_transform_maintains_superbasis(monoclinic_lattice):
    sb = Superbasis.from_pymatgen_lattice(monoclinic_lattice)
    
    for i in range(4):
        sb, _ = sb.selling_transform()
        assert np.isclose(sb.superbasis_vecs[0], -np.sum(sb.superbasis_vecs[1:], axis=0)).all()

def test_can_selling_reduce_vonorm_list(monoclinic_lattice):
    vl = Superbasis.from_pymatgen_lattice(monoclinic_lattice).compute_vonorms()
    tol = 1e-5
    reduced, _ = selling_reduce(vl, tol=tol)
    assert reduced.is_obtuse(tol=tol)

def test_parallel_reduction_rhombohedral(rhombohedral_lattice):

    sb = Superbasis.from_pymatgen_lattice(rhombohedral_lattice)
    vl = sb.compute_vonorms()

    for i in range(100):
        assert np.all(np.isclose(vl.vonorms, sb.compute_vonorms().vonorms))
        sb_temp, _ = sb.selling_transform()
        sb_converged = sb_temp == sb

        if sb_converged:
            print("Superbasis converged!")

        vl_temp, _ = vl.selling_transform()
        vl_converged = vl_temp == vl

        if vl_converged:
            print("Vonorms converged!")
        
        sb = sb_temp
        vl = vl_temp
        print(f"Got through transform {i}")
        if sb_converged and vl_converged:
            break

def test_parallel_reduction_monoclinic(monoclinic_lattice):

    sb = Superbasis.from_pymatgen_lattice(monoclinic_lattice)
    vl = sb.compute_vonorms()

    for i in range(40):
        assert np.all(np.isclose(vl.vonorms, sb.compute_vonorms().vonorms))
        sb_temp, _ = sb.selling_transform()
        sb_converged = sb_temp == sb

        # if sb_converged:
        #     print("Superbasis converged!")

        vl_temp, _ = vl.selling_transform()
        vl_converged = vl_temp == vl

        # if vl_converged:
        #     print("Vonorms converged!")
        
        sb = sb_temp
        vl = vl_temp
        # print(f"Got through transform {i}")
        if sb_converged and vl_converged:
            break

@pytest.mark.parametrize(
    "lattice",
    [
        (Lattice.cubic(1.2)),
        (Lattice.rhombohedral(2.8, 59)),
        (Lattice.hexagonal(5.6, 4.9)),
        (Lattice.monoclinic(5.6, 4.9, 8.1, 110)),
    ]
)
def test_selling_reductions_equivalent(lattice):
    superbasis = Superbasis.from_pymatgen_lattice(lattice)

    # print(f"Vonorms before Selling: {superbasis.compute_vonorms()}")
    # print(f"Conorms before Selling: {superbasis.compute_vonorms().conorms}")
    # print("=============")
    reduced_sb, nsteps = selling_reduce(superbasis, tol=1e-7)
    # print(f"Vonorms after {nsteps} SB Selling steps: {reduced_sb.compute_vonorms()}")
    # print(f"Conorms after {nsteps} SB Selling steps: {reduced_sb.compute_vonorms().conorms}")
    # print("=============")
    reduced_vonorms, nsteps = selling_reduce(superbasis.compute_vonorms())
    # print(f"Vonorms after {nsteps} VO Selling steps: {reduced_vonorms}")
    # print(f"Conorms after {nsteps} VO Selling steps: {reduced_vonorms.conorms}")
    assert reduced_sb.compute_vonorms().has_same_members(reduced_vonorms)