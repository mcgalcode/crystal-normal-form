import pytest
import numpy as np
import helpers
import os
from cnf.cnf_constructor import CNFConstructor
from cnf.navigation.lattice_neighbor_finder import LatticeNeighborFinder
from cnf.lattice.voronoi import VonormList
from pymatgen.core.structure import Structure


@helpers.parameterized_by_mp_structs
def test_batch_canonicalize_equals_individual(idx, struct: Structure):
    """Test that batch canonicalization in Rust produces same results as individual Python canonicalization."""
    xi = 1.5
    delta = 20

    constructor = CNFConstructor(xi, delta, False)
    original_cnf = constructor.from_pymatgen_structure(struct).cnf

    # Get validated step data (same for both)
    lnf = LatticeNeighborFinder(original_cnf.xi, original_cnf.delta, original_cnf.elements)

    # Get raw step data
    if 'USE_RUST' in os.environ:
        del os.environ['USE_RUST']
    step_data = lnf._compute_step_data_raw_python(original_cnf.coords)

    # Validate steps
    validated_steps = []
    for step_vec, vonorms_tuple, motif_coords, matrix in step_data:
        vonorms = VonormList(vonorms_tuple)

        if not vonorms.has_valid_conorms_exact():
            continue
        if not (vonorms.is_obtuse() and vonorms.is_superbasis_exact()):
            continue

        validated_steps.append((step_vec, vonorms_tuple, motif_coords, matrix))

    if len(validated_steps) == 0:
        pytest.skip("No valid steps for this structure")

    print(f"\nStructure {idx}: {len(validated_steps)} validated steps")

    # Python: canonicalize each individually using canonicalize_tuple
    py_constructor = CNFConstructor(xi, delta, verbose_logging=False)
    py_results = []
    atoms = lnf.elements
    atoms_str = [str(atom) for atom in atoms]

    for step_vec, vonorms_tuple, motif_coords, matrix in validated_steps:
        coords_tuple = tuple(motif_coords.flatten())
        cnf_tuple = (*vonorms_tuple, *coords_tuple)
        canonical_tuple = py_constructor.canonicalize_tuple(cnf_tuple, atoms_str)
        py_results.append(canonical_tuple)

    # Rust: batch canonicalize
    from cnf import rust_cnf

    # Convert validated_steps to ensure proper types for Rust
    # Note: Rust expects coords to INCLUDE origin atom at (0,0,0)
    validated_steps_converted = []
    for step_vec, vonorms_tuple, motif_coords, matrix in validated_steps:
        # Prepend origin atom at (0, 0, 0) to coords
        coords_with_origin = np.vstack([np.array([[0, 0, 0]]), motif_coords])
        validated_steps_converted.append((
            step_vec,
            list(vonorms_tuple),  # Convert tuple to list for Rust
            np.ascontiguousarray(coords_with_origin.flatten(), dtype=np.float64),  # Flatten to 1D with origin
            matrix.flatten().tolist()  # Convert matrix to list
        ))

    canonical_results = rust_cnf.canonicalize_cnfs_batch_rust(
        validated_steps_converted,
        atoms_str,
        float(xi),
        int(delta)
    )

    rust_results = []
    for canonical_vonorms_list, canonical_coords_list in canonical_results:
        rust_results.append((
            tuple([int(v) for v in canonical_vonorms_list[:7]]),
            tuple([int(c) for c in canonical_coords_list])
        ))

    print(f"  Python results: {len(py_results)}")
    print(f"  Rust results: {len(rust_results)}")

    # Compare counts
    assert len(py_results) == len(rust_results), (
        f"Different number of results! Python: {len(py_results)}, Rust: {len(rust_results)}"
    )

    # Compare each result
    for i, (py_result, rust_result) in enumerate(zip(py_results, rust_results)):
        py_vonorms = py_result[:7]
        py_motif = py_result[7:]
        
        rust_vonorms, rust_motif = rust_result

        if py_vonorms != rust_vonorms:
            print(f"\n  Result {i}: Vonorms mismatch")
            print(f"    Python: {py_vonorms}")
            print(f"    Rust:   {rust_vonorms}")

        if py_motif != rust_motif:
            print(f"\n  Result {i}: Motif mismatch")
            print(f"    Python (first 12): {py_motif[:12]}")
            print(f"    Rust (first 12):   {rust_motif[:12]}")

        assert py_vonorms == rust_vonorms, f"Result {i}: Vonorms don't match"
        assert py_motif == rust_motif, f"Result {i}: Motif doesn't match"

    print(f"  ✓ All {len(py_results)} canonical results match!")
