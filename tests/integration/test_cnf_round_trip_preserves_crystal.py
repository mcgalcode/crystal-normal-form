from cnf import CrystalNormalForm

from pymatgen.core.structure import Structure
import helpers

@helpers.skip_if_fast
def test_cnf_round_trip_yields_same_crystal_full_cells(mp_structures: list[Structure]):
    xi = 0.01
    delta = 10000


    for struct in mp_structures:
        cnf = CrystalNormalForm.from_pymatgen_structure(struct, xi, delta)
        recovered_struct = cnf.reconstruct()

        helpers.assert_identical_by_pdd_distance(struct, recovered_struct)

@helpers.skip_if_fast
def test_cnf_round_trip_yields_same_crystal_primitive_cells(mp_structures: list[Structure]):
    xi = 0.01
    delta = 10000

    for struct in mp_structures:
        struct = struct.to_primitive()
        cnf = CrystalNormalForm.from_pymatgen_structure(struct, xi, delta)
        recovered_struct = cnf.reconstruct()
        helpers.assert_identical_by_pdd_distance(struct, recovered_struct)
        
