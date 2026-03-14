import pytest
import numpy as np
import helpers

from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.lattice.lnf_constructor import VonormCanonicalizer, LatticeNormalForm
from cnf.navigation.neighbor_finder import NeighborFinder
from pymatgen.core.structure import Structure
from cnf.unit_cell import UnitCell

from pathlib import Path

STRUCT_SAMPLE_FREQ = 1

@helpers.skip_if_fast
@helpers.parameterized_by_mp_structs
def test_cnf_neighbor_reciprocity(idx, struct: Structure):
    verbose = False
    xi = 1.5
    delta = 100
    constructor = CNFConstructor(xi, delta, False) 

    struct = struct.to_primitive()
    original_cnf = constructor.from_pymatgen_structure(struct).cnf

    nf = NeighborFinder.from_cnf(original_cnf)
    nbs = nf.find_neighbors(original_cnf)
    for nb in nbs:
        nb2s = nf.find_neighbors(nb)
        assert original_cnf in nb2s
