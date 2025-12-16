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
from cnf.linalg import MatrixTuple


@helpers.parameterized_by_mp_structs
def test_s4_groups_equal(idx, struct: Structure):
    xi = 1.5
    delta = 20

    before = os.getenv('USE_RUST')

    constructor = CNFConstructor(xi, delta, False)
    original_cnf = constructor.from_pymatgen_structure(struct).cnf
    # PYTHON
    if before is not None:
        del os.environ['USE_RUST']
    nf = NeighborFinder.from_cnf(original_cnf)
    py_nbs = nf.find_neighbors(original_cnf)

    import rust_cnf

    for nb in py_nbs:
        vonorms = nb.lattice_normal_form.vonorms

        # Python S4 groups
        py_s4_groups = vonorms.maximally_ascending_equivalence_class_members()

        # Rust S4 groups
        vonorms_arr = np.array(vonorms.vonorms, dtype=np.float64)
        rust_s4_groups = rust_cnf.get_s4_maximal_representatives_rust(vonorms_arr)

        # Compare number of groups
        assert len(py_s4_groups) == len(rust_s4_groups), (
            f"Different number of S4 groups: Python {len(py_s4_groups)}, Rust {len(rust_s4_groups)}"
        )

        # Compare transition matrices for each group, matching by S4 key
        for group_key, group_data in py_s4_groups.items():
            # group_key is the S4 key (sorted first 4 permutation indices)

            # Find corresponding Rust group by S4 key
            rust_group = None
            for rg in rust_s4_groups:
                if tuple(rg['s4_key']) == group_key:
                    rust_group = rg
                    break

            assert rust_group is not None, f"Couldn't find Rust group for S4 key {group_key}"

            maximal_list = group_data['permuted_vonorms']
            vonorms_tuple = tuple(maximal_list.vonorms)

            # Compare transition matrices
            py_mats = set()
            for trans_mat in group_data['transition_mats']:
                mat_tuple = MatrixTuple(trans_mat.matrix)
                py_mats.add(mat_tuple)

            rust_mats = set()
            for mat in rust_group['transition_matrices']:
                mat_tuple = MatrixTuple(np.array(mat, dtype=np.int32))
                rust_mats.add(mat_tuple)

            assert py_mats == rust_mats, (
                f"Transition matrices don't match for group {vonorms_tuple}!\n"
                f"Python has {len(py_mats)}, Rust has {len(rust_mats)}\n"
                f"Python-only: {py_mats - rust_mats}\n"
                f"Rust-only: {rust_mats - py_mats}"
            )
