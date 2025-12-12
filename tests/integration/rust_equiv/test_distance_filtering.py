"""Test that Rust and Python distance filtering produce identical results."""

import pytest
import numpy as np
import helpers
import os
import rust_cnf
from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.navigation.neighbor_finder import NeighborFinder
from cnf.navigation.utils import filter_out_by_min_dist
from cnf.lattice.voronoi import VonormList
from cnf.lattice.lattice_normal_form import LatticeNormalForm
from cnf.motif.motif_normal_form import MotifNormalForm
from pymatgen.core.structure import Structure


@helpers.parameterized_by_mp_structs
def test_rust_python_distance_filtering_equal(idx, struct: Structure):
    """
    Test that Rust and Python distance filtering produce identical results.

    For each MP structure:
    1. Find all neighbors
    2. Filter using Python implementation
    3. Filter using Rust implementation
    4. Assert the filtered sets are identical
    """
    xi = 1.5
    delta = 20
    min_distance = 1.4  # Angstroms

    # Create CNF from structure
    constructor = CNFConstructor(xi, delta, False)
    original_cnf = constructor.from_pymatgen_structure(struct).cnf

    # Find all neighbors (using Rust if available)
    save_rust_env = os.getenv('USE_RUST')
    os.environ['USE_RUST'] = "1"

    nf = NeighborFinder(original_cnf)
    neighbor_tuples = nf.find_neighbor_tuples()

    # If no neighbors found, skip test
    if len(neighbor_tuples) == 0:
        pytest.skip("No neighbors found for this structure")

    max_neighbors_to_test = 10
    neighbor_tuples = neighbor_tuples[:max_neighbors_to_test]

    # Convert to list format for Rust (avoid intermediate CNF construction)
    neighbor_list = [
        ([int(v) for v in vonorms], [int(c) for c in coords])
        for vonorms, coords in neighbor_tuples
    ]

    n_atoms = len(original_cnf.motif_normal_form.elements)

    # Python filtering using Rust geometry but Python loop
    # This still constructs CNF objects but avoids pymatgen reconstruction
    from cnf.navigation.utils import filter_out_by_min_dist
    

    
    nb_cnfs = []
    for vonorms_tuple, coords_tuple in neighbor_tuples:
        cnf = CrystalNormalForm.from_tuple(tuple([*vonorms_tuple, *coords_tuple]), original_cnf.elements, original_cnf.xi, original_cnf.delta)
        nb_cnfs.append(cnf)

    python_filtered_cnfs = filter_out_by_min_dist(nb_cnfs, min_distance)
    python_filtered_tuples = [cnf.coords for cnf in python_filtered_cnfs]

    # Rust filtering - direct tuple comparison
    rust_filtered_tuples = rust_cnf.filter_neighbors_by_min_distance_rust(
        neighbor_list,
        n_atoms,
        float(xi),
        int(delta),
        min_distance
    )

    # Convert to sets of tuples for comparison (no CNF object construction needed!)
    python_filtered_set = set(python_filtered_tuples)
    rust_filtered_set = set(rust_filtered_tuples)

    # Debug output
    print(f"\nStructure {idx}: {struct.composition.reduced_formula}")
    print(f"Total neighbors: {len(neighbor_tuples)}")
    print(f"Python filtered: {len(python_filtered_set)}")
    print(f"Rust filtered: {len(rust_filtered_set)}")
    print(f"Python only: {len(python_filtered_set - rust_filtered_set)}")
    print(f"Rust only: {len(rust_filtered_set - python_filtered_set)}")
    print(f"Common: {len(python_filtered_set & rust_filtered_set)}")

    # Show examples if there are differences
    if python_filtered_set != rust_filtered_set:
        python_only = list(python_filtered_set - rust_filtered_set)[:3]
        rust_only = list(rust_filtered_set - python_filtered_set)[:3]

        print("\nPython-only neighbors:")
        for t in python_only:
            vonorms_tuple = t[:7]
            coords_tuple = t[7:]
            print(f"  Vonorms: {vonorms_tuple}")
            print(f"  Coords (first 12): {coords_tuple[:12]}")

        print("\nRust-only neighbors:")
        for vonorms_tuple, coords_tuple in rust_only:
            print(f"  Vonorms: {vonorms_tuple}")
            print(f"  Coords (first 12): {coords_tuple[:12]}")

    # Restore environment
    if save_rust_env is not None:
        os.environ['USE_RUST'] = save_rust_env
    elif 'USE_RUST' in os.environ:
        del os.environ['USE_RUST']

    # Assert equality
    assert python_filtered_set == rust_filtered_set, (
        f"Filtered neighbor sets differ: "
        f"Python has {len(python_filtered_set - rust_filtered_set)} extra, "
        f"Rust has {len(rust_filtered_set - python_filtered_set)} extra"
    )
