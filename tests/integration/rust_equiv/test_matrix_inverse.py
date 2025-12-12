"""Test that Rust computes correct matrix inverse"""

import pytest
import numpy as np
import helpers
import rust_cnf

from cnf.cnf_constructor import CNFConstructor
from pymatgen.core.structure import Structure


@helpers.parameterized_by_mp_structs
def test_rust_matrix_inverse(idx, struct: Structure):
    """Test that Rust inverts lattice matrices correctly"""
    xi = 1.5
    delta = 20

    constructor = CNFConstructor(xi, delta, False)
    original_cnf = constructor.from_pymatgen_structure(struct).cnf

    # Get lattice matrix
    py_struct = original_cnf.reconstruct()
    lattice_matrix = py_struct.lattice.matrix

    # Python inverse
    py_inv = np.linalg.inv(lattice_matrix)

    # Rust inverse (test via reconstruct which uses the inverse internally)
    vonorms = np.array(original_cnf.lattice_normal_form.vonorms.vonorms, dtype=np.float64)
    coords = np.array(original_cnf.motif_normal_form.coord_list, dtype=np.int32)
    n_atoms = len(original_cnf.motif_normal_form.elements)

    rust_lattice_flat, _ = rust_cnf.reconstruct_structure_from_cnf(
        vonorms,
        coords,
        n_atoms,
        float(xi),
        int(delta)
    )

    rust_lattice = np.array(rust_lattice_flat).reshape(3, 3)

    # Compute inverse manually to test
    rust_inv = np.linalg.inv(rust_lattice)

    # They should match
    np.testing.assert_allclose(
        py_inv,
        rust_inv,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Inverse matrices don't match"
    )

    # Also verify inverse property: A @ A^-1 = I
    identity_test = rust_lattice @ rust_inv
    np.testing.assert_allclose(
        identity_test,
        np.eye(3),
        rtol=1e-10,
        atol=1e-10,
        err_msg="Matrix inversion doesn't produce identity"
    )
