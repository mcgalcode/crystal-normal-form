import pytest
import numpy as np
import helpers
import os
from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.navigation.lattice_neighbor_finder import LatticeNeighborFinder
from pymatgen.core.structure import Structure


@helpers.parameterized_by_mp_structs
def test_step_data_equal(idx, struct: Structure):
    """Test that Python and Rust generate identical step data."""
    xi = 1.5
    delta = 20

    constructor = CNFConstructor(xi, delta, False)
    original_cnf = constructor.from_pymatgen_structure(struct).cnf

    # Create neighbor finder
    lnf = LatticeNeighborFinder(original_cnf)

    # Get step data from Python
    if 'USE_RUST' in os.environ:
        del os.environ['USE_RUST']
    py_step_data = lnf._compute_step_data_raw_python()

    # Get step data from Rust
    os.environ['USE_RUST'] = '1'
    lnf_rust = LatticeNeighborFinder(original_cnf)
    rust_step_data = lnf_rust._compute_step_data_raw_rust()

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
