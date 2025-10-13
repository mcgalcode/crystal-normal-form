import helpers
import numpy as np
from cnf.lattice.permutations import is_permutation_set_closed
from cnf.unit_cell import UnitCell
from cnf.linalg import MatrixTuple

@helpers.parameterized_by_mp_structs
def test_stabilizers_form_groups(idx, struct):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()

    vonorms = uc.vonorms
    vo_stab = [p.vonorm_permutation for p in vonorms.stabilizer_perms(tol=1e-4)]
    assert is_permutation_set_closed(vo_stab)

    co_stab = [p.conorm_permutation for p in vonorms.stabilizer_perms(tol=1e-4)]
    assert is_permutation_set_closed(co_stab)

    mats = vonorms.stabilizer_matrices(tol=1e-4)
    for mat1 in mats:
        for mat2 in mats:
            # assert mat1 @ MatrixTuple(np.random.randint(1,3, size=(3,3))) in mats
            assert mat1 @ mat2 in mats
    
    for mat in mats:
        assert mat.inverse() in mats



