import pytest
import numpy as np
import helpers
import os
from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.navigation.neighbor_finder import NeighborFinder
from pymatgen.core.structure import Structure


@helpers.parameterized_by_mp_structs
def test_find_neighbor_tuples_equal(idx, struct: Structure):
    """Test that Python and Rust find the same neighbor tuples."""
    xi = 1.5
    delta = 20

    constructor = CNFConstructor(xi, delta, False)
    original_cnf = constructor.from_pymatgen_structure(struct).cnf

    print(f"\nStructure {idx}:")
    print(f"  Vonorms: {original_cnf.lattice_normal_form.vonorms.tuple}")
    print(f"  Coords: {original_cnf.motif_normal_form.coord_list[:6]}...")

    # Get Python neighbors using NeighborFinder
    neighbor_finder = NeighborFinder(original_cnf)
    py_neighbors = neighbor_finder.find_neighbor_tuples()
    print(f"  Python found {len(py_neighbors)} neighbors")

    # Get Rust neighbors using find_neighbor_tuples_rust
    os.environ['USE_RUST'] = '1'
    import rust_cnf

    # Prepare inputs
    vonorms_i32 = np.array(original_cnf.lattice_normal_form.vonorms.tuple, dtype=np.int32)
    coords_i32 = np.array(original_cnf.motif_normal_form.coord_list, dtype=np.int32)

    elements = [str(el) for el in original_cnf.motif_normal_form.elements]
    n_atoms = len(elements)

    rust_neighbors = rust_cnf.find_neighbor_tuples_rust(
        vonorms_i32,
        coords_i32,
        elements,
        n_atoms,
        xi,
        delta
    )
    print(f"  Rust found {len(rust_neighbors)} neighbors")

    # Convert both to sets of tuples for comparison
    py_neighbor_set = set(py_neighbors)
    rust_neighbor_set = set(rust_neighbors)

    # Check counts
    assert len(py_neighbor_set) == len(rust_neighbor_set), (
        f"Neighbor counts don't match!\n"
        f"  Python found {len(py_neighbor_set)} unique neighbors\n"
        f"  Rust found {len(rust_neighbor_set)} unique neighbors"
    )

    # Check that all Python neighbors are in Rust results
    missing_in_rust = py_neighbor_set - rust_neighbor_set
    if missing_in_rust:
        print(f"  Missing in Rust: {len(missing_in_rust)} neighbors")
        for i, neighbor in enumerate(list(missing_in_rust)[:3]):
            print(f"    Example {i+1}: vonorms={neighbor[0]}, coords={neighbor[1][:6]}...")
        assert False, f"Rust is missing {len(missing_in_rust)} neighbors that Python found"

    # Check that all Rust neighbors are in Python results
    extra_in_rust = rust_neighbor_set - py_neighbor_set
    if extra_in_rust:
        print(f"  Extra in Rust: {len(extra_in_rust)} neighbors")
        for i, neighbor in enumerate(list(extra_in_rust)[:3]):
            print(f"    Example {i+1}: vonorms={neighbor[0]}, coords={neighbor[1][:6]}...")
        assert False, f"Rust found {len(extra_in_rust)} extra neighbors that Python didn't find"

    print(f"  ✓ Both Python and Rust found identical neighbors!")
