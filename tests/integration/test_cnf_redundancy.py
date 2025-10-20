import pytest
import helpers
import numpy as np
import tqdm
import os

from pymatgen.core.structure import Structure, Lattice
from cnf.unit_cell import UnitCell
from cnf import CrystalNormalForm
from cnf.motif.mnf_constructor import MNFConstructor, FractionalMotif
from cnf.cnf_constructor import CNFConstructor

import pathlib

from cnf.linalg.unimodular import UNIMODULAR_MATRICES

# Generate all unimodular matrices up to col max norm 4 or 5
# Save those
# For a test case crystal:
# 1) Generate every possible unit cell using these unimodular matrices
#       - those whose vonorm values are not changed
#       - those with determinant == 1 (no rotoinversions)
#   -> Assert that the lattice normal form string is the same for all of these
#   -> Assert that the crystal normal form string is the same
# 2) For every unimodular matrix figure out if it changes the
#    conorm string or the vonorm string (find stabilizers)
# 3) Do quickly (wish list)

verbose = False
delta = 100
xi = 0.001
# There are 642,000 unimodular mats
sample_freq = 2000
num_test_reqs = 100

import numpy as np

@pytest.mark.debug
@helpers.parameterized_by_mp_structs
def test_lnf_is_unique(idx, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()
    original_vonorms = uc.superbasis.compute_vonorms()
    assert original_vonorms.is_obtuse(tol=1e-5)
    assert uc.is_obtuse(tol=1e-5)
    helpers.printif(f"Considering struct of class V{uc.voronoi_class}", verbose)
    lnfs = []
    for u in UNIMODULAR_MATRICES[::sample_freq]:
        other_cell = uc.apply_unimodular(u)
        if helpers.are_geo_matches(uc, other_cell, tol=1e-4):
            cnf = other_cell.to_cnf(xi, delta)
            lnfs.append(cnf.lattice_normal_form)

    assert len(lnfs) > num_test_reqs
    assert len(set(lnfs)) == 1

@pytest.mark.debug
@helpers.parameterized_by_mp_structs
def test_mnf_is_unique(idx, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()

    original_vonorms = uc.superbasis.compute_vonorms()
    assert original_vonorms.is_obtuse(tol=1e-5)
    assert uc.is_obtuse(tol=1e-5)
    helpers.printif(f"Considering struct of class V{uc.voronoi_class}", verbose)
    
    mnfs = []
    lnfs = []
    for u in UNIMODULAR_MATRICES[::sample_freq]:
        other_cell = uc.apply_unimodular(u)
        if helpers.are_geo_matches(uc, other_cell, tol=1e-4):
            cnf = other_cell.to_cnf(xi, delta)
            mnfs.append(tuple(cnf.motif_normal_form.coord_list))
            lnfs.append(cnf.lattice_normal_form)

    assert len(lnfs) > num_test_reqs
    if len(set(lnfs)) == 1:
        assert len(set(mnfs)) == 1

@helpers.parameterized_by_mp_structs
def test_cnf_is_unique(idx, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()
    helpers.printif(f"Considering struct of class V{uc.voronoi_class}", verbose)
    
    cnf_map: dict[CrystalNormalForm, list] = {}
    all_cnfs: list[CrystalNormalForm] = []
    for u in UNIMODULAR_MATRICES[::sample_freq]:
        other_cell = uc.apply_unimodular(u)
        match, reason = helpers.are_unit_cells_geo_matches(uc, other_cell, tol=1e-4)
        if match:
            cnf = other_cell.to_cnf(xi, delta)
            all_cnfs.append(cnf)
            if cnf.coords in cnf_map:
                cnf_map[cnf.coords].append(other_cell)
            else:
                cnf_map[cnf.coords] = [other_cell]
    
    assert len(all_cnfs) > 100
    cnfs = list(cnf_map.keys())
    assert len(set(cnfs)) == 1

@pytest.mark.debug
@helpers.parameterized_by_mp_structs
def test_undiscretized_mnf_unique(idx, struct: Structure):
    # struct = struct.to_primitive()
    uc = UnitCell.from_pymatgen_structure(struct).reduce()
    all_cells: list[UnitCell] = []
    mats = UNIMODULAR_MATRICES[::sample_freq]
    for u in mats:
        other_cell = uc.apply_unimodular(u).reduce()
        if helpers.are_geo_matches(uc, other_cell):
            all_cells.append(other_cell)

    assert len(all_cells) > 100
    assert len(set([c.vonorms for c in all_cells])) > 10
    # assert len(set([c.motif.to_mnf_list() for c in all_cells])) > 
    con = CNFConstructor(1.5, 20)
    cnfs = [con.from_vonorms_and_motif(cell.vonorms, cell.motif).cnf for cell in all_cells]
    for start_idx in range(0, len(cnfs[0].motif_normal_form.coord_list) - 10, 10):
        all_coord_lists = sorted(cnfs, key = lambda cnf: cnf.motif_normal_form.coord_list)
        pieces = [a.motif_normal_form.coord_list[start_idx:start_idx+10] for a in all_coord_lists]
        if len(set(pieces)) > 1:
            for cnf in pieces:
                print(cnf)
            print()
            print()
    assert len(set([c.motif_normal_form.coord_list for c in cnfs])) == 1