import pytest
import numpy as np
import helpers

from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.lattice.lnf_constructor import VonormCanonicalizer, LatticeNormalForm
from cnf.navigation.neighbor_finder import NeighborFinder
from pymatgen.core.structure import Structure
from cnf.unit_cell import UnitCell


@helpers.parameterized_by_mp_structs
def test_neighbors_are_unique(idx, struct: Structure):
    verbose = False
    save_pairs = False
    xi = 1.5
    delta = 20
    constructor = CNFConstructor(xi, delta, False)
    original_cnf = constructor.from_pymatgen_structure(struct).cnf
    nf = NeighborFinder.from_cnf(original_cnf)
    nbs = nf.find_neighbors(original_cnf)
    unique_nbs = set(nbs)
    assert len(nbs) == len(unique_nbs)
    assert original_cnf not in nbs

