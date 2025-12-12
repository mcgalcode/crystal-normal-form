import pytest
import numpy as np
import helpers
import os
from cnf.cnf_constructor import CNFConstructor
from cnf.linalg import MatrixTuple
from pymatgen.core.structure import Structure
import rust_cnf


@helpers.parameterized_by_mp_structs
def test_permuted_vonorm_stabilizers_equal(idx, struct: Structure):
    """Test that stabilizers of permuted vonorms match between Python and Rust during step generation."""
    xi = 1.5
    delta = 20

    constructor = CNFConstructor(xi, delta, False)
    original_cnf = constructor.from_pymatgen_structure(struct).cnf

    vonorms = original_cnf.lattice_normal_form.vonorms

    # Get Python S4 groups
    py_s4_groups = vonorms.maximally_ascending_equivalence_class_members()

    # Get Rust S4 groups
    vonorms_arr = np.array(vonorms.vonorms, dtype=np.float64)
    rust_s4_groups = rust_cnf.get_s4_maximal_representatives_rust(vonorms_arr)

    print(f"\nStructure {idx}: {len(py_s4_groups)} S4 groups")

    # For each S4 group, compare stabilizers of the permuted vonorms
    for s4_key, py_data in py_s4_groups.items():
        # Find matching Rust group by S4 key
        rust_group = None
        for rg in rust_s4_groups:
            if tuple(rg['s4_key']) == s4_key:
                rust_group = rg
                break

        assert rust_group is not None, f"Couldn't find Rust group for S4 key {s4_key}"

        # Get permuted vonorms
        py_permuted_vonorms = py_data['maximal_permuted_list']
        rust_permuted_vonorms_arr = np.array(rust_group['maximal_permuted_vonorms'])

        # Verify the permuted vonorms themselves match
        np.testing.assert_array_almost_equal(
            np.array(py_permuted_vonorms.vonorms),
            rust_permuted_vonorms_arr,
            decimal=10,
            err_msg=f"Permuted vonorms don't match for S4 key {s4_key}"
        )

        # Get stabilizers of the permuted vonorms
        py_stabilizers = set(py_permuted_vonorms.stabilizer_matrices())

        rust_stabilizers_raw = rust_cnf.find_stabilizers_rust(rust_permuted_vonorms_arr)
        rust_stabilizers = set(MatrixTuple(s) for s in rust_stabilizers_raw)

        # Compare
        if py_stabilizers != rust_stabilizers:
            print(f"  S4 key {s4_key}:")
            print(f"    Permuted vonorms: {py_permuted_vonorms.vonorms}")
            print(f"    Python stabilizers: {len(py_stabilizers)}")
            print(f"    Rust stabilizers: {len(rust_stabilizers)}")
            print(f"    Python-only: {py_stabilizers - rust_stabilizers}")
            print(f"    Rust-only: {rust_stabilizers - py_stabilizers}")

        assert py_stabilizers == rust_stabilizers, (
            f"Stabilizers of permuted vonorms don't match for S4 key {s4_key}!\n"
            f"  Permuted vonorms: {py_permuted_vonorms.vonorms}\n"
            f"  Python has {len(py_stabilizers)}, Rust has {len(rust_stabilizers)}"
        )
