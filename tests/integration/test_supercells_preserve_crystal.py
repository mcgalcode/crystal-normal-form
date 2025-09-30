import pytest
import os

from cnf import CrystalNormalForm
from cnf.unit_cell import UnitCell

from pymatgen.core.structure import Structure

import helpers

@helpers.skip_if_fast
def test_supercell_construction_preserves_crystal_structure(mp_structures: list[Structure]):
    MAX_IDX = 4
    for cell_idx in range(2, MAX_IDX + 1):
        for struct in mp_structures[::100]:
            supercells = UnitCell.from_pymatgen_structure(struct).supercells(cell_idx)
            for cell in supercells:
                superstruct = cell.to_pymatgen_structure()
                helpers.assert_identical_by_pdd_distance(struct, superstruct)

@helpers.skip_if_fast
def test_supercell_construction_preserves_crystal_structure_after_cnf(mp_structures: list[Structure]):
    xi = 0.001
    delta = 1000
    MAX_IDX = 4

    for cell_idx in range(2, MAX_IDX + 1):
        for struct in mp_structures[::100]:
            supercells = UnitCell.from_pymatgen_structure(struct).supercells(cell_idx)
            for cell in supercells:                
                superstruct = cell.to_pymatgen_structure()
                cnf = CrystalNormalForm.from_pymatgen_structure(superstruct, xi, delta)
                recovered_struct = cnf.reconstruct()
                helpers.assert_identical_by_pdd_distance(struct, recovered_struct, cutoff=0.05)