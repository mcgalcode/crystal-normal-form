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
def test_neighbors_are_unique(idx, struct: Structure):
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
        py_stab = nb.lattice_normal_form.vonorms.stabilizer_matrices_fast()

        rust_stab = rust_cnf.find_stabilizers_rust(np.array(nb.lattice_normal_form.vonorms.vonorms, dtype=np.float64))
        rust_stab = [MatrixTuple(s) for s in rust_stab]
        assert len(py_stab) >= 1
        assert set(py_stab) == set(rust_stab)
        