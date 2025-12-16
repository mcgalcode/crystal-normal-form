import pytest
import numpy as np
import helpers
import os
from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.navigation.neighbor_finder import NeighborFinder
from cnf.navigation.motif_neighbor_finder import extract_coord_matrix_from_mnf_tuple
from pymatgen.core.structure import Structure


@helpers.parameterized_by_mp_structs
def test_validate_step_data_equal(idx, struct: Structure):
    """Test that Python and Rust validate step data identically."""
    xi = 1.5
    delta = 20

    constructor = CNFConstructor(xi, delta, False)
    original_cnf = constructor.from_pymatgen_structure(struct).cnf

    # Create neighbor finder
    nf = NeighborFinder.from_cnf(original_cnf)

    # Get step data from Rust (unvalidated)
    os.environ['USE_RUST'] = '1'
    import rust_cnf

    vonorms = original_cnf.lattice_normal_form.vonorms
    input_vonorms = np.array(vonorms.vonorms, dtype=np.float64)

    # Extract motif data from MNF
    motif_coord_matrix, n_atoms = extract_coord_matrix_from_mnf_tuple(original_cnf.coords[7:], include_origin=True)
    motif_coords_flat = np.ascontiguousarray(motif_coord_matrix.flatten(), dtype=np.float64)

    # Note: Stabilizers are computed internally in Rust
    step_data = rust_cnf.compute_step_data_raw_rust(
        input_vonorms, motif_coords_flat,
        n_atoms, delta
    )

    print(f"\nStructure {idx}:")
    print(f"  Total step data entries: {len(step_data)}")

    # Validate using Python method
    py_validated = nf.lattice_neighbor_finder.validate_step_data(step_data)
    print(f"  Python validated: {len(py_validated)}")

    # Validate using Rust function
    rust_validated = rust_cnf.validate_step_data_rust(step_data)
    print(f"  Rust validated: {len(rust_validated)}")

    # First just check counts
    assert len(py_validated) == len(rust_validated), (
        f"Validation counts don't match!\n"
        f"  Python validated {len(py_validated)} steps\n"
        f"  Rust validated {len(rust_validated)} steps"
    )
