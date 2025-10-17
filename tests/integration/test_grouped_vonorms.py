import helpers
from helpers.data import mp_structs_with_voronoi_class

from cnf.unit_cell import UnitCell

def test_v2_groups():
    xi = 1.5
    delta = 20
    for struct in mp_structs_with_voronoi_class(5)[:10]:
        cell = UnitCell.from_pymatgen_structure(struct)
        cnf = cell.to_cnf(xi, delta)

        eq_class_members = cnf.lattice_normal_form.vonorms.maximally_ascending_equivalence_class_members()
        eqms = [(group, data['maximal_permuted_list'], len(data['stabilizing_mats'])) for group, data in eq_class_members.items()]
        print(eqms)