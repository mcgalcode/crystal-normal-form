import numpy as np

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from cnf.lattice.superbasis import Superbasis
from cnf.lattice.selling import SuperbasisSellingReducer
from cnf.lattice.selling.selling_pair import SellingPair
from cnf.lattice.selling.selling_transform_matrix import SellingTransformMatrix

def test_unimodular_inverse():
    for p in SellingPair.CANONICAL_PAIRS:
        mat = SellingTransformMatrix.from_pair(p).matrix
        inv = SellingTransformMatrix.inverse_from_pair(p).matrix
        assert np.all(mat @ inv == np.eye(3))

def test_selling_transform_lattice_vecs():
    lat = Lattice.from_parameters(1.0, 1.5, 1.25, 50, 70, 120)
    sb = Superbasis.from_pymatgen_lattice(lat)
    reducer = SuperbasisSellingReducer()
    new_sb, swap = reducer.apply_selling_transform(sb)
    transform_matrix = SellingTransformMatrix.from_pair(swap)
    # print("Original generators")
    # print(sb.generating_vecs())

    # print("transformed generators")
    # print(new_sb.generating_vecs())

    # print(transform_matrix)
    assert ((sb.generating_vecs().T @ transform_matrix.matrix).T == new_sb.generating_vecs()).all()

def test_selling_transform_atom_positions():
    lat = Lattice.from_parameters(1.0, 1.5, 1.25, 50, 70, 120)
    struct = Structure(lat, ["Li", "Li"], [(0.25, 0.25, 0.25), (0.5, 0.5, 0)])

    fractional_coords = struct.frac_coords.T
    original_cart_coords = struct.lattice.matrix.T @ fractional_coords
    assert np.all(original_cart_coords == struct.cart_coords.T)

    sb = Superbasis.from_pymatgen_lattice(lat)
    NUM_ITERATIONS = 10
    transformed_fractional_coords = np.copy(fractional_coords)
    reducer = SuperbasisSellingReducer()
    for _ in range(NUM_ITERATIONS):
        sb, swap = reducer.apply_selling_transform(sb)
        transform_matrix = SellingTransformMatrix.from_pair(swap)
        transformed_fractional_coords = (transform_matrix.matrix @ transformed_fractional_coords)
        transformed_cart_coords = sb.generating_vecs().T @ transformed_fractional_coords
        assert np.all(np.isclose(transformed_cart_coords, original_cart_coords))

def test_selling_reduce_atomic_positions():
    lat = Lattice.from_parameters(1.0, 1.5, 1.25, 50, 70, 120)
    struct = Structure(lat, ["Li", "Li"], [(0.25, 0.25, 0.25), (0.5, 0.5, 0)])

    fractional_coords = struct.frac_coords.T
    original_cart_coords = struct.cart_coords.T

    sb = Superbasis.from_pymatgen_lattice(lat)
    reducer = SuperbasisSellingReducer(tol=1e-7)
    reduce_result = reducer.reduce(sb)
    reduced_sb = reduce_result.reduced_object
    print(reduce_result.all_transform_matrices)
    transform_mat = reduce_result.transform_matrix
    reduced_frac_coords = transform_mat.inverse() @ fractional_coords
    reduced_cart_coords = reduced_sb.generating_vecs().T @ reduced_frac_coords

    assert np.all(np.isclose(reduced_cart_coords, original_cart_coords))

