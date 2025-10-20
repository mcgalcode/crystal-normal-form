import helpers
import pytest
import numpy as np

from cnf.lattice import Superbasis
from cnf.lattice.selling import SuperbasisSellingReducer, VonormListSellingReducer, SellingTransformMatrix

from pymatgen.core.lattice import Lattice
from cnf.unit_cell import UnitCell

def test_can_selling_transform(monoclinic_lattice):
    basis = Superbasis.from_pymatgen_lattice(monoclinic_lattice)
    vonorm_list = basis.compute_vonorms()
    sb_reducer = SuperbasisSellingReducer()
    vl_reducer = VonormListSellingReducer()

    transformed_basis, sb_swap = sb_reducer.apply_selling_transform(basis)
    # print(transformed_basis.compute_vonorms())

    transformed_vonorms, v_swap = vl_reducer.apply_selling_transform(basis.compute_vonorms())
    assert sb_swap == v_swap
    # print(f"Vonorm swap: {v_swap}")
    # print(f"Superbasis swap: {sb_swap}")
    # print(vonorm_list)
    # print(transformed_vonorms)

    assert np.isclose(transformed_basis.compute_vonorms().vonorms, transformed_vonorms.vonorms).all()

def test_reduction_pair_selection_is_the_same(acute_lattice_1):
    basis = Superbasis.from_pymatgen_lattice(acute_lattice_1)
    vonorm_list = basis.compute_vonorms()
    sb_reducer = SuperbasisSellingReducer(tol=1e-3)
    vl_reducer = VonormListSellingReducer(tol=1e-3, verbose_logging=False)


    num_steps = 500
    for i in range(num_steps):
        basis, sb_swap = sb_reducer.apply_selling_transform(basis)
        vonorm_list, v_swap = vl_reducer.apply_selling_transform(vonorm_list)
        assert basis.is_obtuse() == vonorm_list.is_obtuse()
        assert sb_swap == v_swap
        assert np.isclose(basis.compute_vonorms().vonorms, vonorm_list.vonorms).all()

        if basis.is_obtuse(tol=1e-3):
            print(f"Got obtuseness after {i} steps")
            break
    
    # print(basis.compute_vonorms().conorms)
    # print(vonorm_list.conorms)

def test_parallel_reduction_rhombohedral(rhombohedral_lattice):

    sb = Superbasis.from_pymatgen_lattice(rhombohedral_lattice)
    vl = sb.compute_vonorms()

    sb_reducer = SuperbasisSellingReducer()
    vl_reducer = VonormListSellingReducer()

    for i in range(100):
        assert vl.has_same_members(sb.compute_vonorms())
        # assert np.all(np.isclose(vl.vonorms, sb.compute_vonorms().vonorms))
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
        sb_temp, sb_selected_pair = sb_reducer.apply_selling_transform(sb)
        sb_converged = sb_temp == sb

        # if sb_converged:
        #     print("Superbasis converged!")

        vl_temp, vl_selected_pair = vl_reducer.apply_selling_transform(vl)
        vl_converged = vl_temp == vl

        assert sb_selected_pair == vl_selected_pair
        # if vl_converged:
        #     print("Vonorms converged!")
        
        sb = sb_temp
        vl = vl_temp
        # print(f"Got through transform {i}")
        if sb_converged and vl_converged:
            break

def test_reduction_mats_equal(monoclinic_lattice):

    sb = Superbasis.from_pymatgen_lattice(monoclinic_lattice)
    vl = sb.compute_vonorms()

    sb_reducer = SuperbasisSellingReducer()
    vl_reducer = VonormListSellingReducer()    

    for i in range(40):
        assert np.all(np.isclose(vl.vonorms, sb.compute_vonorms().vonorms))
        sb_temp, sb_selected_pair = sb_reducer.apply_selling_transform(sb)
        sb_converged = sb_temp == sb

        # if sb_converged:
        #     print("Superbasis converged!")

        vl_temp, vl_selected_pair = vl_reducer.apply_selling_transform(vl)
        vl_converged = vl_temp == vl

        assert sb_selected_pair == vl_selected_pair
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

@helpers.parameterized_by_mp_struct_idxs([446])
def test_steps_maintain_structure(idx, struct):
    uc = UnitCell.from_pymatgen_structure(struct)
    helpers.assert_identical_by_pdd_distance(struct, uc.to_pymatgen_structure(), 0.000001)

    sb = uc.superbasis
    vl = uc.vonorms
    motif = uc.motif

    sb_reducer = SuperbasisSellingReducer(tol=1e-3)
    # vl_reducer = VonormListSellingReducer()

    for i in range(10):
        if sb.is_obtuse():
            break
        sb, sb_pair = sb_reducer.apply_selling_transform(sb)
        print()
        print(sb.compute_vonorms().conorms)
        print(sb.compute_vonorms())
        mat = SellingTransformMatrix.from_pair(sb_pair)
        if mat.determinant() == -1:
            mat = mat.flip_signs()
            sb = sb_reducer.apply_sign_flip_to_object(sb)        

        motif = motif.apply_unimodular(mat)
        new_uc = UnitCell(sb, motif)
        new_uc.to_cif(f"selling_step_{i}_struct_{idx}.cif")
        reduced = new_uc.to_pymatgen_structure()

        pdd = helpers.pdd(struct, reduced)
        print(f"PDD was {pdd} after {i} steps.")
