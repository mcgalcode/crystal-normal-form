import pytest
import numpy as np
import helpers
import os
from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.motif.mnf_constructor import extract_coord_matrix_from_mnf_tuple
from cnf.navigation.lattice_neighbor_finder import LatticeNeighborFinder
from pymatgen.core.structure import Structure


def _compute_step_data_raw_rust(cnf_tuple: tuple, delta):
    """Rust implementation of step data computation."""
    import rust_cnf

    # Get motif data using helper (Rust needs origin atom)
    motif_coord_matrix, n_atoms = extract_coord_matrix_from_mnf_tuple(cnf_tuple[7:], include_origin=True)

    # Prepare input vonorms - Rust will compute S4 groups and stabilizers internally
    input_vonorms = np.array(cnf_tuple[:7], dtype=np.float64)

    # Call Rust function - Rust now handles stabilizers and S4 grouping internally!
    motif_coords_flat = np.ascontiguousarray(motif_coord_matrix.flatten(), dtype=np.float64)
    result = rust_cnf.compute_step_data_raw_rust(
        input_vonorms,
        motif_coords_flat,
        n_atoms,
        delta
    )

    # Convert result back to Python format
    # Result is list of (step_vec, vonorms_tuple, coords, matrix_flat)
    # Note: coords from Rust include origin, need to remove it to match Python format
    python_result = []
    for step_vec, vonorms_list, coords_flat, mat_flat in result:
        vonorms_tuple = tuple([int(v) for v in vonorms_list])
        # Rust returns coords WITH origin, Python expects WITHOUT origin
        coords_with_origin = np.array(coords_flat).reshape(n_atoms, 3)
        coords = coords_with_origin[1:]  # Remove origin atom
        matrix = np.array(mat_flat, dtype=np.int32).reshape(3, 3)
        python_result.append((step_vec, vonorms_tuple, coords, matrix))

    return python_result

@helpers.parameterized_by_mp_structs
def test_step_data_equal(idx, struct: Structure):
    """Test that Python and Rust generate identical step data."""
    xi = 1.5
    delta = 20

    constructor = CNFConstructor(xi, delta, False)
    original_cnf = constructor.from_pymatgen_structure(struct).cnf

    # Create neighbor finder
    lnf = LatticeNeighborFinder(xi, delta, original_cnf.elements)

    # Get step data from Python
    if 'USE_RUST' in os.environ:
        del os.environ['USE_RUST']
    py_step_data = lnf._compute_step_data_raw_python(original_cnf.coords)

    rust_step_data = _compute_step_data_raw_rust(original_cnf.coords, delta)

    print(f"\nStructure {idx}:")
    print(f"  Python step data entries: {len(py_step_data)}")
    print(f"  Rust step data entries: {len(rust_step_data)}")

    # Convert to comparable format - include ALL components
    # Sort the step data to compare content regardless of order
    py_steps = []
    for step_vec, vonorms_tuple, coords, matrix in py_step_data:
        # Create sortable key from ALL step data components
        key = (
            tuple(step_vec),  # Step vector
            vonorms_tuple,     # Vonorms
            tuple(coords.flatten()),  # Flattened motif coords
            tuple(matrix.flatten())   # Flattened transformation matrix
        )
        py_steps.append(key)

    rust_steps = []
    for step_vec, vonorms_tuple, coords, matrix in rust_step_data:
        # Ensure matrix is numpy array for consistent flattening
        matrix_arr = np.array(matrix) if not isinstance(matrix, np.ndarray) else matrix
        key = (
            tuple(step_vec),
            vonorms_tuple,
            tuple(coords.flatten()),
            tuple(matrix_arr.flatten())
        )
        rust_steps.append(key)

    # Sort both lists for order-independent comparison
    py_steps_sorted = sorted(py_steps)
    rust_steps_sorted = sorted(rust_steps)

    # Check if they match when sorted
    if py_steps_sorted == rust_steps_sorted:
        print(f"  ✓ Step data matches (same content, possibly different order)")
    else:
        # If sorted lists don't match, show set-based comparison
        py_steps_set = set(py_steps)
        rust_steps_set = set(rust_steps)
        py_only = py_steps_set - rust_steps_set
        rust_only = rust_steps_set - py_steps_set
        common = py_steps_set & rust_steps_set

        print(f"  Common steps: {len(common)}")
        print(f"  Python-only steps: {len(py_only)}")
        print(f"  Rust-only steps: {len(rust_only)}")

        if py_only or rust_only:
            print("\n  Python-only steps (first 5):")
            for i, (step_vec, vonorms, coords, matrix) in enumerate(list(py_only)[:5]):
                print(f"    {i}:")
                print(f"      step_vec: {step_vec}")
                print(f"      vonorms: {vonorms}")
                print(f"      matrix: {np.array(matrix).reshape(3, 3)}")

            print("\n  Rust-only steps (first 5):")
            for i, (step_vec, vonorms, coords, matrix) in enumerate(list(rust_only)[:5]):
                print(f"    {i}:")
                print(f"      step_vec: {step_vec}")
                print(f"      vonorms: {vonorms}")
                print(f"      matrix: {np.array(matrix).reshape(3, 3)}")

    assert py_steps_sorted == rust_steps_sorted, (
        f"Step data doesn't match even when sorted!\n"
        f"  Python generated {len(py_step_data)} steps ({len(set(py_steps))} unique)\n"
        f"  Rust generated {len(rust_step_data)} steps ({len(set(rust_steps))} unique)\n"
        f"  This indicates actual differences in content, not just ordering."
    )
