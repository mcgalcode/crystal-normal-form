import pytest
import os

from cnf.cnf_constructor import CNFConstructor
from cnf.unit_cell import UnitCell

from pymatgen.core.structure import Structure

import helpers

@helpers.skip_if_fast
@helpers.parameterized_by_mp_structs
def test_supercell_construction_preserves_crystal_structure(idx, struct: Structure):
    MAX_IDX = 4
    for cell_idx in range(2, MAX_IDX + 1):
        supercells = UnitCell.from_pymatgen_structure(struct).supercells(cell_idx)
        for cell in supercells:
            superstruct = cell.to_pymatgen_structure()
            helpers.assert_identical_by_pdd_distance(struct, superstruct)

@helpers.skip_if_fast
@helpers.parameterized_by_mp_structs
def test_supercell_construction_preserves_crystal_structure_after_cnf(idx: int, struct: Structure):
    xi = 0.00001
    delta = 100000
    MAX_IDX = 3
    constructor = CNFConstructor(xi, delta)
    for cell_idx in range(2, MAX_IDX + 1):
        supercells = UnitCell.from_pymatgen_structure(struct).supercells(cell_idx)
        idx_delta = delta * cell_idx
        constructor = CNFConstructor(xi, idx_delta)
        for cell in supercells:                
            superstruct = cell.to_pymatgen_structure()
            cnf = constructor.from_pymatgen_structure(superstruct).cnf
            recovered_struct = cnf.reconstruct()
            helpers.assert_identical_by_pdd_distance(struct, recovered_struct, cutoff=0.001)