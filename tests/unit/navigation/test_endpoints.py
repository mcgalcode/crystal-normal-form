import pytest

from pymatgen.core.structure import Structure
from cnf.unit_cell import UnitCell
from cnf.utils.pdd import pdd
from cnf.navigation.endpoints import calculate_supercell_indices, \
                                     normalize_endpoint, \
                                     are_endpoints_compatible, \
                                     get_endpoint_cnfs, \
                                     get_endpoint_unit_cells

def test_calc_supercell_indices_simple():

    sa, ea = calculate_supercell_indices(1,2)
    assert sa == 2
    assert ea == 1

    sa, ea = calculate_supercell_indices(3,2)
    assert sa == 2
    assert ea == 3

    sa, ea = calculate_supercell_indices(3,3)
    assert sa == 1
    assert ea == 1

def test_calc_supercell_indices_min_atoms():
    sa, ea = calculate_supercell_indices(1,2, min_atoms=4)
    assert sa == 4
    assert ea == 2

    sa, ea = calculate_supercell_indices(2,2, min_atoms=7)
    assert sa == 4
    assert ea == 4

    sa, ea = calculate_supercell_indices(3,5, min_atoms=23)
    assert sa == 10
    assert ea == 6

def test_normalize_endpoint(zr_bcc_manual_unit_cell):
    xi = 1.0
    delta = 10

    endpt_cnf = zr_bcc_manual_unit_cell.to_cnf(xi, delta)
    endpt_struct = zr_bcc_manual_unit_cell.to_pymatgen_structure()
    endpt_uc = zr_bcc_manual_unit_cell

    normal_endpt = normalize_endpoint(endpt_cnf)
    assert isinstance(normal_endpt, UnitCell)
    assert normal_endpt.to_cnf(xi, delta) == endpt_cnf 

    normal_endpt = normalize_endpoint(endpt_struct)
    assert isinstance(normal_endpt, UnitCell)
    assert normal_endpt.to_cnf(xi, delta) == endpt_cnf

    normal_endpt = normalize_endpoint(endpt_uc)
    assert isinstance(normal_endpt, UnitCell)
    assert normal_endpt.to_cnf(xi, delta) == endpt_cnf

def test_are_endpts_compatible(zr_bcc_manual_unit_cell, zr_fcc_manual_unit_cell):

    assert are_endpoints_compatible(zr_bcc_manual_unit_cell, zr_fcc_manual_unit_cell)
    assert are_endpoints_compatible(zr_bcc_manual_unit_cell.to_pymatgen_structure(),
                                    zr_fcc_manual_unit_cell.to_pymatgen_structure())
    assert are_endpoints_compatible(zr_bcc_manual_unit_cell,
                                    zr_fcc_manual_unit_cell.to_pymatgen_structure())
    
    assert are_endpoints_compatible(zr_bcc_manual_unit_cell.to_cnf(1,10),
                                    zr_fcc_manual_unit_cell)
    
    assert are_endpoints_compatible(zr_bcc_manual_unit_cell,
                                    zr_fcc_manual_unit_cell.to_cnf(1,10))
    
    assert not are_endpoints_compatible(zr_bcc_manual_unit_cell.to_cnf(1,10),
                                        zr_fcc_manual_unit_cell.to_cnf(2,10))
    
    assert not are_endpoints_compatible(zr_bcc_manual_unit_cell.to_cnf(2,11),
                                        zr_fcc_manual_unit_cell.to_cnf(2,10))
    
    assert are_endpoints_compatible(zr_bcc_manual_unit_cell.to_cnf(2,11),
                                    zr_fcc_manual_unit_cell.to_cnf(2,11))

def test_get_endpoint_unit_cells(zr_bcc_mp, zr_hcp_mp):
    zr_bcc_mp = zr_bcc_mp.to_primitive()
    assert len(zr_bcc_mp) == 1
    assert len(zr_hcp_mp) == 2
    uc1s, uc2s = get_endpoint_unit_cells(zr_hcp_mp, zr_bcc_mp)

    for uc in uc1s + uc2s:
        assert isinstance(uc, UnitCell)
        assert len(uc) == 2
    
    for uc in uc1s:
        assert pdd(zr_hcp_mp, uc.to_pymatgen_structure()) == 0
    
    for uc in uc2s:
        assert pdd(zr_bcc_mp, uc.to_pymatgen_structure()) == 0

    uc1s, uc2s = get_endpoint_unit_cells(zr_hcp_mp, zr_bcc_mp, min_atoms=4)

    for uc in uc1s + uc2s:
        assert isinstance(uc, UnitCell)
        assert len(uc) == 4
    
    for uc in uc1s:
        assert pdd(zr_hcp_mp, uc.to_pymatgen_structure()) == 0
    
    for uc in uc2s:
        assert pdd(zr_bcc_mp, uc.to_pymatgen_structure()) == 0

def test_get_endpoint_cnfs(zr_bcc_mp, zr_hcp_mp):
    zr_bcc_mp = zr_bcc_mp.to_primitive()

    with pytest.raises(ValueError) as ex:
        cnf1s, cnf2s = get_endpoint_cnfs(zr_bcc_mp, zr_hcp_mp)
    assert "Must provide either [xi and delta]" in ex.value.__repr__()

    xi = 1
    delta = 10
    zr_bcc_cnf = UnitCell.from_pymatgen_structure(zr_bcc_mp).to_cnf(xi, delta)

    cnf1s, cnf2s = get_endpoint_cnfs(zr_bcc_cnf, zr_hcp_mp)
    for c in cnf1s + cnf2s:
        assert len(c.reconstruct()) == 2
        assert c.xi == xi
        assert c.delta == delta
    