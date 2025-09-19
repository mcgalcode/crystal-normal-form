import pytest
import numpy as np

from cnf.unit_cell import UnitCell
from cnf.lattice import Superbasis
from cnf.motif import FractionalMotif
from cnf.crystal_normal_form import CrystalNormalForm
from cnf.lattice.lattice_normal_form import LatticeNormalForm
from cnf.lattice.lnf_constructor import LatticeNormalFormConstructor

def test_bcc_zr_unit_cells(zr_bcc_primitive_lattice_vecs):
    sb = Superbasis.from_generating_vecs(zr_bcc_primitive_lattice_vecs)
    motif = FractionalMotif.from_elements_and_positions(["Zr"], [[0, 0, 0]])
    unit_cell = UnitCell(sb, motif)
    
    xi = 1.5
    delta = 30

    supercells = unit_cell.supercells(2)

    lnf_constructor = LatticeNormalFormConstructor(xi)
    unique_lnfs = set([lnf_constructor.build_lnf_from_superbasis(cell.superbasis).lnf for cell in supercells])
    # for cnf in unique_lnfs:
    #     print(cnf.coords)
    assert len(unique_lnfs) == 2

    unique_cnfs = set([CrystalNormalForm.from_unit_cell(cell, xi, delta) for cell in supercells])
    for cnf in unique_cnfs:
        print(cnf.coords)
    assert len(unique_cnfs) == 2

def test_fcc_zr_unit_cells(zr_fcc_primitive_lattice_vecs):
    sb = Superbasis.from_generating_vecs(zr_fcc_primitive_lattice_vecs)
    motif = FractionalMotif.from_elements_and_positions(["Zr"], [[0, 0, 0]])
    unit_cell = UnitCell(sb, motif)
    
    xi = 1.5
    delta = 30

    supercells = unit_cell.supercells(2)
    unique_cnfs = set([CrystalNormalForm.from_unit_cell(cell, xi, delta) for cell in supercells])

    assert len(unique_cnfs) == 2
    for cnf in unique_cnfs:
        print(cnf.coords)