import pytest
import numpy as np

from cnf.lattice import Superbasis
from cnf.lattice.selling import SuperbasisSellingReducer, VonormListSellingReducer

from pymatgen.core.lattice import Lattice

def test_can_selling_transform(monoclinic_lattice):
    basis = Superbasis.from_pymatgen_lattice(monoclinic_lattice)
    vonorm_list = basis.compute_vonorms()
    sb_reducer = SuperbasisSellingReducer()
    vl_reducer = VonormListSellingReducer()

    transformed_basis, swap = sb_reducer.apply_selling_transform(basis)
    print(transformed_basis.compute_vonorms())

    transformed_vonorms, _ = vl_reducer.apply_selling_transform(basis.compute_vonorms())
    print("VIA VONORM")
    print(vonorm_list)
    print(transformed_vonorms)

    assert np.isclose(transformed_basis.compute_vonorms().vonorms, transformed_vonorms.vonorms).all()

def test_parallel_reduction_rhombohedral(rhombohedral_lattice):

    sb = Superbasis.from_pymatgen_lattice(rhombohedral_lattice)
    vl = sb.compute_vonorms()

    sb_reducer = SuperbasisSellingReducer()
    vl_reducer = VonormListSellingReducer()

    for i in range(100):
        assert np.all(np.isclose(vl.vonorms, sb.compute_vonorms().vonorms))
        sb_temp, _ = sb_reducer.apply_selling_transform(sb)
        sb_converged = sb_temp == sb

        if sb_converged:
            print("Superbasis converged!")

        vl_temp, _ = vl_reducer.apply_selling_transform(vl)
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

    sb_reducer = SuperbasisSellingReducer()
    vl_reducer = VonormListSellingReducer()    

    for i in range(40):
        assert np.all(np.isclose(vl.vonorms, sb.compute_vonorms().vonorms))
        sb_temp, _ = sb_reducer.apply_selling_transform(sb)
        sb_converged = sb_temp == sb

        # if sb_converged:
        #     print("Superbasis converged!")

        vl_temp, _ = vl_reducer.apply_selling_transform(vl)
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
        (Lattice.rhombohedral(7.8, 36)),
        (Lattice.rhombohedral(2.8, 59)),
        (Lattice.from_parameters(5.7, 4.3, 2.9, 85, 110, 67)),
        (Lattice.from_parameters(9.2, 2.8, 4.7, 35, 98, 113)),
    ]
)
def test_selling_reductions_equivalent(lattice):
    superbasis = Superbasis.from_pymatgen_lattice(lattice)
    sb_reducer = SuperbasisSellingReducer(tol=1e-7)
    vl_reducer = VonormListSellingReducer()  
    print("========NEXT LATTICE========")
    print(f"Vonorms before Selling: {superbasis.compute_vonorms()}")
    print(f"Conorms before Selling: {superbasis.compute_vonorms().conorms}")
    print("=============")
    sb_reduction_result = sb_reducer.reduce(superbasis)
    print(f"Vonorms after {sb_reduction_result.num_steps} SB Selling steps: {sb_reduction_result.reduced_object.compute_vonorms()}")
    print(f"Conorms after {sb_reduction_result.num_steps} SB Selling steps: {sb_reduction_result.reduced_object.compute_vonorms().conorms}")
    print("*************")
    vl_reduction_result = vl_reducer.reduce(superbasis.compute_vonorms())
    print(f"Vonorms after {vl_reduction_result.num_steps} VO Selling steps: {vl_reduction_result.reduced_object}")
    print(f"Conorms after {vl_reduction_result.num_steps} VO Selling steps: {vl_reduction_result.reduced_object.conorms}")
    assert sb_reduction_result.reduced_object.compute_vonorms().is_superbasis()
    assert vl_reduction_result.reduced_object.is_superbasis()
    assert sb_reduction_result.reduced_object.compute_vonorms().has_same_members(vl_reduction_result.reduced_object)