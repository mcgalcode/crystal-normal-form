import pytest
import helpers

from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.unit_cell import UnitCell
from pymatgen.core.composition import Element

def test_sandbox():
    # xi = 1.5
    # delta = 30
    # els = [Element("Re"), Element("In"), Element("Ge")]
    # cnf1 = CrystalNormalForm.from_tuple(
    #     (13, 13, 13, 13, 13, 13, 26, 15, 7, 22, 15, 22, 7),
    #     els,
    #     xi,
    #     delta
    # )

    # cnf2 = CrystalNormalForm.from_tuple( 
    #     (13, 13, 13, 13, 13, 13, 26, 14, 7, 22, 14, 22, 7),
    #     els,
    #     xi,
    #     delta
    # )

    cnf1_path = helpers.data.get_data_file_path("cnf_pairs/ex1/cnf1.json")
    cnf2_path = helpers.data.get_data_file_path("cnf_pairs/ex1/cnf2.json")

    cnf1 = CrystalNormalForm.from_file(cnf1_path)
    cnf2 = CrystalNormalForm.from_file(cnf2_path)

    xi = cnf1.xi
    delta = cnf1.delta
    print()
    print(f"CNF 1")
    print(f"    LNF: {cnf1.lattice_normal_form.coords}")
    print(f"    BNF: {cnf1.basis_normal_form.coord_list}")
    cnf1.basis_normal_form.to_discretized_motif().print_details()
    print()
    print(f"CNF 2")
    print(f"    LNF: {cnf2.lattice_normal_form.coords}")
    print(f"    BNF: {cnf2.basis_normal_form.coord_list}")
    cnf2.basis_normal_form.to_discretized_motif().print_details()

    uc1 = UnitCell.from_cnf(cnf1)
    uc2 = UnitCell.from_cnf(cnf2)

    s1 = uc1.to_pymatgen_structure()
    s2 = uc2.to_pymatgen_structure()
    pdd = helpers.pdd(s1, s2)

    print(f"Structures have PDD dist of {pdd}")

    assert set(cnf1.lattice_normal_form.vonorms.stabilizer_matrices()) == set(cnf2.lattice_normal_form.vonorms.stabilizer_matrices())
    
    print("Searching for paths from CNF1 to CNF2")
    print(f"Found {len(cnf1.lattice_normal_form.vonorms.stabilizer_perms())} Stabilizer Perms for CNF1 of BNF1")
    print(f"Found {len(cnf1.lattice_normal_form.vonorms.stabilizer_matrices())} Stabilizer Mats for CNF1 of BNF1")

    for s in cnf1.lattice_normal_form.vonorms.stabilizer_matrices():
        bnfl = cnf1.basis_normal_form.to_discretized_motif().apply_unimodular(s).to_bnf_list()
        bnfl = [int(i) for i in bnfl][3:]
        print(s, "->", bnfl)

    print()
    print("Searching for paths from CNF2 to CNF2")
    print(f"Found {len(cnf2.lattice_normal_form.vonorms.stabilizer_perms())} Stabilizer Perms for CNF2 of BNF2")
    print(f"Found {len(cnf2.lattice_normal_form.vonorms.stabilizer_matrices())} Stabilizer Mats for CNF2 of BNF2")

    for s in cnf2.lattice_normal_form.vonorms.stabilizer_matrices():
        bnfl = cnf2.basis_normal_form.to_discretized_motif().apply_unimodular(s).to_bnf_list()
        bnfl = [int(i) for i in bnfl][3:]
        print(s, "->", bnfl)



    recovered_cnf1 = uc1.to_cnf(1.0, delta, verbose=False)
    recovered_cnf2 = uc2.to_cnf(1.0, delta, verbose=False)
    assert recovered_cnf1 == recovered_cnf2

    con = CNFConstructor(xi, delta)
    res = con.from_discretized_obtuse_vonorms_and_motif(cnf1.lattice_normal_form.vonorms, cnf1.basis_normal_form.to_discretized_motif())
    print(res.cnf)