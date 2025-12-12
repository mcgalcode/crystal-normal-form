import pytest
import numpy as np
import helpers
import os
from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.lattice.lnf_constructor import VonormCanonicalizer, LatticeNormalForm
from cnf.navigation.neighbor_finder import NeighborFinder
from cnf.navigation.motif_neighbor_finder import MotifNeighborFinder
from cnf.navigation.lattice_neighbor_finder import LatticeNeighborFinder
from pymatgen.core.structure import Structure
from cnf.unit_cell import UnitCell


@helpers.parameterized_by_mp_structs
def test_neighbors_are_unique(idx, struct: Structure):
    verbose = False
    save_pairs = False
    xi = 1.5
    delta = 20

    before = os.getenv('USE_RUST')

    constructor = CNFConstructor(xi, delta, False)
    original_cnf = constructor.from_pymatgen_structure(struct).cnf
    # PYTHON
    if before is not None:
        del os.environ['USE_RUST']
    nf = NeighborFinder(original_cnf)
    py_nbs = nf.find_neighbors()
    py_lat_nbs = set(LatticeNeighborFinder(original_cnf).find_cnf_neighbors())
    py_mot_nbs = set([n.point for n in MotifNeighborFinder(original_cnf).find_motif_neighbors().neighbors])
    py_unique_nbs = set(py_nbs)

    os.environ['USE_RUST'] = "1"
    constructor = CNFConstructor(xi, delta, False)
    nf = NeighborFinder(original_cnf)
    rust_nbs = nf.find_neighbors()
    rust_lat_nbs = set(LatticeNeighborFinder(original_cnf).find_cnf_neighbors())
    rust_mot_nbs = set([n.point for n in MotifNeighborFinder(original_cnf).find_motif_neighbors().neighbors])
    rust_unique_nbs = set(rust_nbs)

    # Debug output
    print(f"\nPython: {len(py_nbs)} total, {len(py_unique_nbs)} unique")
    print(f"Rust: {len(rust_nbs)} total, {len(rust_unique_nbs)} unique")
    print(f"Python only: {len(py_unique_nbs - rust_unique_nbs)}")
    print(f"Rust only: {len(rust_unique_nbs - py_unique_nbs)}")
    print(f"Common: {len(py_unique_nbs & rust_unique_nbs)}")

    # Check if stabilizers differ for any neighbors
    print("\nChecking stabilizers for mismatched neighbors...")
    py_only = list(py_unique_nbs - rust_unique_nbs)[:3]  # Check first 3
    rust_only = list(rust_unique_nbs - py_unique_nbs)[:3]

    print("\nPython-only neighbors:")
    for cnf in py_only:
        vonorms = cnf.lattice_normal_form.vonorms
        motif = cnf.motif_normal_form.coord_list
        py_stabs = vonorms.stabilizer_matrices_fast()

        # Compute Rust stabilizers
        import rust_cnf
        rust_stabs_flat = rust_cnf.find_stabilizers_rust(np.array(vonorms.vonorms, dtype=np.float64))
        n_rust = len(rust_stabs_flat)  # Returns list of matrices, not flat array

        print(f"\n  Vonorms: {vonorms.vonorms}")
        print(f"  Motif (first 12): {motif[:12]}")
        print(f"  Python stabilizers: {len(py_stabs)}, Rust stabilizers: {n_rust}")

    print("\nRust-only neighbors:")
    for cnf in rust_only:
        vonorms = cnf.lattice_normal_form.vonorms
        motif = cnf.motif_normal_form.coord_list
        py_stabs = vonorms.stabilizer_matrices_fast()

        rust_stabs_flat = rust_cnf.find_stabilizers_rust(np.array(vonorms.vonorms, dtype=np.float64))
        n_rust = len(rust_stabs_flat)

        print(f"\n  Vonorms: {vonorms.vonorms}")
        print(f"  Motif (first 12): {motif[:12]}")
        print(f"  Python stabilizers: {len(py_stabs)}, Rust stabilizers: {n_rust}")


    assert rust_mot_nbs == py_mot_nbs
    assert rust_lat_nbs == py_lat_nbs
    assert py_unique_nbs == rust_unique_nbs
    assert len(rust_nbs) == len(py_nbs)

    os.environ['USE_RUST'] = str(before)
