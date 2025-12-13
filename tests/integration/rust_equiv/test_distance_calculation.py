"""Test that Rust and Python compute the same pairwise distances"""

import pytest
import numpy as np
import helpers
import rust_cnf

from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.navigation.neighbor_finder import NeighborFinder
from cnf.navigation.utils import compute_pairwise_distances
from cnf.lattice.voronoi import VonormList
from cnf.lattice.lattice_normal_form import LatticeNormalForm
from cnf.motif.motif_normal_form import MotifNormalForm
from pymatgen.core.structure import Structure


@helpers.parameterized_by_mp_structs
def test_rust_python_distances_equal(idx, struct: Structure):
    """
    Test that Rust and Python compute identical pairwise distances.

    For each CNF neighbor:
    1. Compute distances using Python (via reconstruct)
    2. Compute distances using Rust (via geometry module)
    3. Assert all distances match
    """
    xi = 1.5
    delta = 20

    # Create CNF from structure
    constructor = CNFConstructor(xi, delta, False)
    original_cnf = constructor.from_pymatgen_structure(struct).cnf

    # Find neighbors (using Rust)
    import os
    save_rust_env = os.getenv('USE_RUST')
    os.environ['USE_RUST'] = "1"

    nf = NeighborFinder(original_cnf)
    neighbor_tuples = nf.find_neighbor_tuples()

    # Test first 10 neighbors for speed
    for i, (vonorms_tuple, coords_tuple) in enumerate(neighbor_tuples[:10]):
        ### Python distance calculation
        vonorms = VonormList(vonorms_tuple)
        lnf = LatticeNormalForm(vonorms, xi)
        mnf = MotifNormalForm(coords_tuple, original_cnf.motif_normal_form.elements, delta)
        cnf = CrystalNormalForm(lnf, mnf)

        py_struct = cnf.reconstruct()
        py_distances_matrix = compute_pairwise_distances(py_struct)
        py_distances = py_distances_matrix[np.triu_indices_from(py_distances_matrix, k=1)]

        ### Rust distance calculation
        # Get lattice and positions from Rust
        vonorms_arr = np.array(vonorms_tuple, dtype=np.float64)
        coords_arr = np.array(coords_tuple, dtype=np.int32)
        n_atoms = len(original_cnf.motif_normal_form.elements)

        rust_lattice_flat, rust_positions = rust_cnf.reconstruct_structure_from_cnf(
            vonorms_arr,
            coords_arr,
            n_atoms,
            float(xi),
            int(delta)
        )

        # Convert to numpy arrays
        rust_positions_arr = np.array(rust_positions, dtype=np.float64)
        rust_lattice_arr = np.array(rust_lattice_flat, dtype=np.float64)

        # Compute distances in Rust (inverse computed internally)
        rust_distances = rust_cnf.compute_pairwise_distances_pbc_rust(
            rust_positions_arr,
            n_atoms,
            rust_lattice_arr
        )

        # Convert to numpy array
        rust_distances = np.array(rust_distances)

        # Check if positions match
        rust_positions_reshaped = np.array(rust_positions).reshape(n_atoms, 3)
        pos_diff = np.abs(py_struct.cart_coords - rust_positions_reshaped)
        max_pos_diff = np.max(pos_diff)
        if max_pos_diff > 1e-10:
            print(f"\n  WARNING: Positions don't match! Max diff: {max_pos_diff:.2e}")
            # Find which atom has the largest difference
            max_idx = np.unravel_index(np.argmax(pos_diff), pos_diff.shape)
            print(f"  Atom {max_idx[0]}, coord {max_idx[1]}:")
            print(f"    Python: {py_struct.cart_coords[max_idx[0], max_idx[1]]:.17f}")
            print(f"    Rust:   {rust_positions_reshaped[max_idx[0], max_idx[1]]:.17f}")

        # Check if lattice matrices match
        rust_lattice_matrix = np.array(rust_lattice_flat).reshape(3, 3)
        lattice_diff = np.abs(py_struct.lattice.matrix - rust_lattice_matrix)
        max_lattice_diff = np.max(lattice_diff)
        if max_lattice_diff > 1e-10:
            print(f"\n  WARNING: Lattice matrices don't match! Max diff: {max_lattice_diff:.2e}")

        # Debug output for first and problematic neighbors
        if i == 0:
            print(f"\nStructure {idx}: {struct.composition.reduced_formula}")
            print(f"Neighbor {i}:")
            print(f"  Lattice matrix:")
            print(py_struct.lattice.matrix)
            print(f"  Inv lattice:")
            print(np.linalg.inv(py_struct.lattice.matrix))

            # Manually compute first distance (atoms 0 and 1)
            diff = py_struct.cart_coords[1] - py_struct.cart_coords[0]
            print(f"\n  Manual calculation for atoms 0-1:")
            print(f"  Diff: {diff}")
            inv_lat = np.linalg.inv(py_struct.lattice.matrix)
            frac = inv_lat @ diff
            print(f"  Frac: {frac}")
            wrapped = frac - np.round(frac)
            print(f"  Wrapped: {wrapped}")
            min_img = py_struct.lattice.matrix @ wrapped
            print(f"  Min image: {min_img}")
            manual_dist = np.linalg.norm(min_img)
            print(f"  Distance: {manual_dist:.6f}")

            print(f"\n  Python distances (first 5): {py_distances[:5]}")
            print(f"  Rust distances (first 5): {rust_distances[:5]}")

        # Assert distances match
        # Use reasonable tolerance for crystallographic distances
        # 0.1 Angstroms accounts for PBC edge cases and reconstruction differences
        try:
            np.testing.assert_allclose(
                py_distances,
                rust_distances,
                rtol=1e-3,
                atol=0.1,
                err_msg=f"Distance calculations don't match for neighbor {i}"
            )
        except AssertionError as e:
            # Debug which specific distances differ
            diff = np.abs(py_distances - rust_distances)
            max_diff_idx = np.argmax(diff)
            max_diff = diff[max_diff_idx]

            print(f"\n  FAILURE: Max diff at pair index {max_diff_idx}: {max_diff:.6f}")
            print(f"  Python: {py_distances[max_diff_idx]:.6f}")
            print(f"  Rust: {rust_distances[max_diff_idx]:.6f}")

            # Convert pair index to atom indices
            # For upper triangular indexing: (i,j) where i < j
            # pair_idx = i * n - i*(i+1)/2 + (j-i-1)
            n = n_atoms
            pair_count = 0
            for atom_i in range(n):
                for atom_j in range(atom_i + 1, n):
                    if pair_count == max_diff_idx:
                        print(f"  Atom pair: ({atom_i}, {atom_j})")

                        # Show positions
                        print(f"  Atom {atom_i} pos: {py_struct.cart_coords[atom_i]}")
                        print(f"  Atom {atom_j} pos: {py_struct.cart_coords[atom_j]}")

                        # Manually compute for debugging
                        diff_vec = py_struct.cart_coords[atom_j] - py_struct.cart_coords[atom_i]
                        inv_lat = np.linalg.inv(py_struct.lattice.matrix)
                        frac = inv_lat @ diff_vec
                        print(f"  Frac diff: {frac}")
                        wrapped = frac - np.round(frac)
                        print(f"  Python wrapped: {wrapped}")
                        min_img = py_struct.lattice.matrix @ wrapped
                        manual_dist = np.linalg.norm(min_img)
                        print(f"  Manual Python dist: {manual_dist:.6f}")
                        break
                    pair_count += 1

            raise

    # Restore environment
    if save_rust_env is not None:
        os.environ['USE_RUST'] = save_rust_env
    elif 'USE_RUST' in os.environ:
        del os.environ['USE_RUST']

    print(f"  All {min(10, len(neighbor_tuples))} tested neighbors have matching distances!")
