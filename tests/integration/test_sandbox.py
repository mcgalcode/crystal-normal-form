import pytest
import helpers

from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.unit_cell import UnitCell
from pymatgen.core.composition import Element

def test_sandbox():
    xi = 1.5
    delta = 30
    els = [Element("Re"), Element("In"), Element("Ge")]
    cnf1 = CrystalNormalForm.from_tuple(
        (13, 13, 13, 13, 13, 13, 26, 15, 7, 22, 15, 22, 7),
        els,
        xi,
        delta
    )

    cnf2 = CrystalNormalForm.from_tuple( 
        (13, 13, 13, 13, 13, 13, 26, 14, 7, 22, 14, 22, 7),
        els,
        xi,
        delta
    )

    print(f"CNF 1")
    print(f"LNF: {cnf1.lattice_normal_form.coords}")
    print(f"BNF: {cnf1.basis_normal_form.coord_list}")
    print(cnf1.basis_normal_form.to_discretized_motif().print_details())

    print(f"CNF 2")
    print(f"LNF: {cnf2.lattice_normal_form.coords}")
    print(f"BNF: {cnf2.basis_normal_form.coord_list}")
    print(cnf2.basis_normal_form.to_discretized_motif().print_details())

    uc1 = UnitCell.from_cnf(cnf1)
    uc2 = UnitCell.from_cnf(cnf2)

    s1 = uc1.to_pymatgen_structure()
    s2 = uc2.to_pymatgen_structure()
    pdd = helpers.pdd(s1, s2)

    print(f"Structures have PDD dist of {pdd}")

    assert set(cnf1.lattice_normal_form.vonorms.stabilizer_matrices()) == set(cnf2.lattice_normal_form.vonorms.stabilizer_matrices())
    
    print("Possible transformations of BNF1")
    # for perm in cnf1.lattice_normal_form.vonorms.stabilizer_perms():
    #     for s in perm.all_matrices:
    for s in cnf1.lattice_normal_form.vonorms.stabilizer_matrices():
        bnfl = cnf1.basis_normal_form.to_discretized_motif().apply_unimodular(s).to_bnf_list()
        bnfl = [int(i) for i in bnfl][3:]
        if tuple(bnfl) == cnf2.basis_normal_form.coord_list:
            print(bnfl)
            # print(cnf1.lattice_normal_form.vonorms.apply_permutation(perm.vonorm_permutation))

    recovered_cnf1 = uc1.to_cnf(xi, delta, verbose=False)
    recovered_cnf2 = uc2.to_cnf(xi, delta, verbose=False)
    assert recovered_cnf1 == recovered_cnf2

    con = CNFConstructor(xi, delta)
    res = con.from_discretized_obtuse_vonorms_and_motif(cnf1.lattice_normal_form.vonorms, cnf1.basis_normal_form.to_discretized_motif())
    print(res.cnf)