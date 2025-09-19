import numpy as np

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from cnf.lattice.superbasis import Superbasis
from cnf.lattice.utils import selling_reduce
from cnf.lattice.selling import CANONICAL_PAIRS, SELLING_TRANSFORM_INVERSE_MATRICES, SELLING_TRANSFORM_MATRICES

def test_unimodular_inverse():
    for p in CANONICAL_PAIRS:
        mat = SELLING_TRANSFORM_MATRICES[p]
        inv = SELLING_TRANSFORM_INVERSE_MATRICES[p]
        assert np.all(mat @ inv == np.eye(3))

def test_selling_transform_lattice_vecs():
    lat = Lattice.from_parameters(1.0, 1.5, 1.25, 50, 70, 120)
    sb = Superbasis.from_pymatgen_lattice(lat)
    new_sb, swap = sb.selling_transform()
    transform_matrix = SELLING_TRANSFORM_MATRICES[swap]
    # print("Original generators")
    # print(sb.generating_vecs())

    # print("transformed generators")
    # print(new_sb.generating_vecs())

    # print(transform_matrix)
    assert ((sb.generating_vecs().T @ transform_matrix).T == new_sb.generating_vecs()).all()

def test_selling_transform_atom_positions():
    lat = Lattice.from_parameters(1.0, 1.5, 1.25, 50, 70, 120)
    struct = Structure(lat, ["Li", "Li"], [(0.25, 0.25, 0.25), (0.5, 0.5, 0)])

    fractional_coords = struct.frac_coords.T
    original_cart_coords = struct.lattice.matrix.T @ fractional_coords
    assert np.all(original_cart_coords == struct.cart_coords.T)

    sb = Superbasis.from_pymatgen_lattice(lat)
    NUM_ITERATIONS = 10
    transformed_fractional_coords = np.copy(fractional_coords)
    for _ in range(NUM_ITERATIONS):
        sb, swap = sb.selling_transform()
        transform_matrix = SELLING_TRANSFORM_INVERSE_MATRICES[swap]
        transformed_fractional_coords = (transform_matrix @ transformed_fractional_coords)
        transformed_cart_coords = sb.generating_vecs().T @ transformed_fractional_coords
        assert np.all(np.isclose(transformed_cart_coords, original_cart_coords))

def test_selling_reduce_atomic_positions():
    lat = Lattice.from_parameters(1.0, 1.5, 1.25, 50, 70, 120)
    struct = Structure(lat, ["Li", "Li"], [(0.25, 0.25, 0.25), (0.5, 0.5, 0)])

    fractional_coords = struct.frac_coords.T
    original_cart_coords = struct.cart_coords.T

    sb = Superbasis.from_pymatgen_lattice(lat)
    reduced_sb, num_steps, transform_mat = selling_reduce(sb, tol=1e-7, return_transform_mat=True)
    reduced_frac_coords = np.linalg.inv(transform_mat) @ fractional_coords
    reduced_cart_coords = reduced_sb.generating_vecs().T @ reduced_frac_coords

    assert np.all(np.isclose(reduced_cart_coords, original_cart_coords))

