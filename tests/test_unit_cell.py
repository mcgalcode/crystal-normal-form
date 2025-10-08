import pytest

import helpers

from cnf.unit_cell import UnitCell
from cnf.lattice import Superbasis
from cnf.motif import FractionalMotif
from cnf.cnf_constructor import CNFConstructor
from cnf.lattice.lnf_constructor import LatticeNormalFormConstructor

from pymatgen.core.structure import Structure

def test_bcc_zr_unit_cells(zr_bcc_primitive_lattice_vecs):
    sb = Superbasis.from_generating_vecs(zr_bcc_primitive_lattice_vecs)
    motif = FractionalMotif.from_elements_and_positions(["Zr"], [[0, 0, 0]])
    unit_cell = UnitCell(sb, motif)
    
    xi = 1.5
    delta = 30

    supercells = unit_cell.supercells(2)
    print(f"Considering {len(supercells)} distinct supercells")

    lnf_constructor = LatticeNormalFormConstructor(xi)
    unique_lnfs = set([lnf_constructor.build_lnf_from_superbasis(cell.superbasis).lnf for cell in supercells])
    for cnf in unique_lnfs:
        print(cnf.coords)
    assert len(unique_lnfs) == 2

    cnfs = []
    cnf_constructor = CNFConstructor(xi, delta, False)
    for cell in supercells:
        # print(f"Putting cell into CNF: {cell.superbasis.compute_vonorms()}")
        cnf = cnf_constructor.from_unit_cell(cell).cnf
        cnfs.append(cnf)
        # print()
        
    unique_cnfs = set(cnfs)    
    
    # for cnf in unique_cnfs:
    #     print(cnf.coords)
    assert len(unique_cnfs) == 2

def test_fcc_zr_unit_cells(zr_fcc_primitive_lattice_vecs):
    sb = Superbasis.from_generating_vecs(zr_fcc_primitive_lattice_vecs)
    motif = FractionalMotif.from_elements_and_positions(["Zr"], [[0, 0, 0]])
    unit_cell = UnitCell(sb, motif)
    
    xi = 1.5
    delta = 30

    supercells = unit_cell.supercells(2)
    lnf_constructor = LatticeNormalFormConstructor(xi)
    unique_lnfs = set([lnf_constructor.build_lnf_from_superbasis(cell.superbasis).lnf for cell in supercells])
    assert len(unique_lnfs) == 2

    cnfs = []
    constructor = CNFConstructor(xi, delta, False)
    for cell in supercells:
        # print(f"Putting cell into CNF: {cell.superbasis.compute_vonorms()}")
        cnf = constructor.from_unit_cell(cell).cnf
        cnfs.append(cnf)
        # print()

    unique_cnfs = set(cnfs)

    assert len(unique_cnfs) == 2
    for cnf in unique_cnfs:
        print(cnf.coords)

STRUCT_SAMPLE_FREQ = 10

@pytest.mark.parametrize("idx, struct", enumerate(helpers.ALL_MP_STRUCTURES[::STRUCT_SAMPLE_FREQ]))
def test_unit_cell_doesnt_change_struct(idx: int, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct)
    pmg_2 = uc.to_pymatgen_structure()
    helpers.assert_identical_by_pdd_distance(struct, pmg_2, cutoff=1e-9)

@pytest.mark.parametrize("idx, struct", enumerate(helpers.ALL_MP_STRUCTURES[::STRUCT_SAMPLE_FREQ]))
def test_reduced_unit_cell_doesnt_change_struct(idx: int, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()
    pmg_2 = uc.to_pymatgen_structure()
    helpers.assert_identical_by_pdd_distance(struct, pmg_2, cutoff=1e-7)