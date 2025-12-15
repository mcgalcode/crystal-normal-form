import pytest
import numpy as np
import helpers
import rust_cnf

from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.lattice.lnf_constructor import VonormCanonicalizer, LatticeNormalForm
from cnf.navigation.neighbor_finder import NeighborFinder
from cnf.navigation.neighbor import Neighbor
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from cnf.unit_cell import UnitCell

from pathlib import Path

STRUCT_SAMPLE_FREQ = 1

@helpers.skip_if_fast
@helpers.parameterized_by_mp_structs
def test_rust_reconstructs_lattice_correctly(idx, struct: Structure):
    """
    Test that Rust correctly reconstructs lattice matrices from vonorms.

    For each CNF neighbor:
    1. Get Python reconstruction (lattice + coordinates)
    2. Get Rust reconstruction (using vonorms_to_lattice_matrix)
    3. Assert lattices match
    4. Assert atomic coordinates match
    """
    verbose = False
    xi = 1.5
    delta = 100
    constructor = CNFConstructor(xi, delta, False)

    struct = struct.to_primitive()
    original_cnf = constructor.from_pymatgen_structure(struct).cnf

    nbs = NeighborFinder(original_cnf).find_neighbors()

    # Test every 10th neighbor for speed
    for nb in nbs[::10]:
        ### Get the Python reconstruction
        py_struct = nb.reconstruct()
        py_lattice_matrix = py_struct.lattice.matrix
        py_cart_coords = py_struct.cart_coords

        ### Get Rust reconstruction
        # Extract vonorms and coords from CNF
        vonorms = np.array(nb.lattice_normal_form.vonorms.vonorms, dtype=np.float64)
        coords = np.array(nb.motif_normal_form.coord_list, dtype=np.int32)
        n_atoms = len(nb.motif_normal_form.elements)

        # Call Rust functions to reconstruct
        rust_lattice_matrix, rust_cart_coords = rust_cnf.reconstruct_structure_from_cnf(
            vonorms,
            coords,
            n_atoms,
            float(xi),
            int(delta)
        )

        # Convert to numpy arrays for comparison
        rust_lattice_matrix = np.array(rust_lattice_matrix).reshape(3, 3)  # Reshape from flat to 3x3
        rust_cart_coords = np.array(rust_cart_coords).reshape(n_atoms, 3)

        # Assert lattice matrices match (within tolerance)
        np.testing.assert_allclose(
            py_lattice_matrix,
            rust_lattice_matrix,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Lattice matrices don't match for neighbor {nb}"
        )

        # Assert coordinates match (within tolerance)
        np.testing.assert_allclose(
            py_cart_coords,
            rust_cart_coords,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Cartesian coordinates don't match for neighbor {nb}"
        )

    if verbose:
        print(f"Structure {idx}: {struct.composition.reduced_formula} - All {len(nbs[::10])} tested neighbors match!")

