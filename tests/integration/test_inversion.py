import numpy as np
import pytest
import helpers

from cnf.unit_cell import UnitCell

@helpers.parameterized_by_mp_struct_idxs(every=100)
def test_inverted_motifs_are_inverted(idx, struct):
    uc = UnitCell.from_pymatgen_structure(struct)
    inverted = uc.motif.transform(-np.eye(3))
    assert inverted.find_inverted_match(uc.motif)