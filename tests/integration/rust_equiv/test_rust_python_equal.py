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
def test_rust_and_python_recover_same_neighbors(idx, struct: Structure):
    xi = 1.5
    delta = 20

    before = os.getenv('USE_RUST')

    constructor = CNFConstructor(xi, delta, False)
    original_cnf = constructor.from_pymatgen_structure(struct).cnf
    original_cnf_tup = original_cnf.coords
    # PYTHON
    if before is not None:
        del os.environ['USE_RUST']
    nf = NeighborFinder.from_cnf(original_cnf)
    py_nbs = nf.find_neighbors(original_cnf)
    py_lat_nbs = nf.find_lattice_neighbors(original_cnf_tup)
    py_mot_nbs = nf.find_motif_neighbors(original_cnf_tup)
    py_unique_nbs = set(py_nbs)

    os.environ['USE_RUST'] = "1"
    constructor = CNFConstructor(xi, delta, False)
    nf = NeighborFinder.from_cnf(original_cnf)
    rust_nbs = nf.find_neighbors(original_cnf)
    rust_lat_nbs = nf.find_lattice_neighbors(original_cnf_tup)
    rust_mot_nbs = nf.find_motif_neighbors(original_cnf_tup)
    rust_unique_nbs = set(rust_nbs)

    # Debug output
    print("================ ALL NEIGHBORS ===============")
    print(f"\nPython: {len(py_nbs)} total, {len(py_unique_nbs)} unique")
    print(f"Rust: {len(rust_nbs)} total, {len(rust_unique_nbs)} unique")
    print(f"Python only: {len(py_unique_nbs - rust_unique_nbs)}")
    print(f"Rust only: {len(rust_unique_nbs - py_unique_nbs)}")
    print(f"Common: {len(py_unique_nbs & rust_unique_nbs)}")

    print("================ LATTICE NEIGHBORS ===============")
    print(f"\nPython: {len(py_lat_nbs)} total, {len(set(py_lat_nbs))} unique")
    print(f"Rust: {len(rust_lat_nbs)} total, {len(set(rust_lat_nbs))} unique")
    print(f"Python only: {len(set(py_lat_nbs) - set(rust_lat_nbs))}")
    print(f"Rust only: {len(set(rust_lat_nbs) - set(py_lat_nbs))}")
    print(f"Common: {len(set(py_lat_nbs) & set(rust_lat_nbs))}")

    print("================ MOTIF NEIGHBORS ===============")
    print(f"\nPython: {len(py_mot_nbs)} total, {len(set(py_mot_nbs))} unique")
    print(f"Rust: {len(rust_mot_nbs)} total, {len(set(rust_mot_nbs))} unique")
    print(f"Python only: {len(set(py_mot_nbs) - set(rust_mot_nbs))}")
    print(f"Rust only: {len(set(rust_mot_nbs) - set(py_mot_nbs))}")
    print(f"Common: {len(set(py_mot_nbs) & set(rust_mot_nbs))}")

    # Check if stabilizers differ for any neighbors
    print("\nChecking stabilizers for mismatched neighbors...")
    py_only = list(py_unique_nbs - rust_unique_nbs)[:3]  # Check first 3
    rust_only = list(rust_unique_nbs - py_unique_nbs)[:3]

    from cnf import rust_cnf

    print("\nPython-only neighbors:")
    for cnf in py_only:
        vonorms = cnf.lattice_normal_form.vonorms
        motif = cnf.motif_normal_form.coord_list
        py_stabs = vonorms.stabilizer_matrices_fast()

        # Compute Rust stabilizers
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


    assert set(rust_mot_nbs) == set(py_mot_nbs)
    assert len(rust_mot_nbs) == len(py_mot_nbs)
    
    assert set(rust_lat_nbs) == set(py_lat_nbs)
    assert len(rust_lat_nbs) == len(py_lat_nbs)

    assert py_unique_nbs == rust_unique_nbs
    assert len(rust_nbs) == len(py_nbs)

    os.environ['USE_RUST'] = str(before)
