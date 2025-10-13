import pytest
import helpers
import numpy as np
import tqdm
import os

from pymatgen.core.structure import Structure, Lattice
from cnf.unit_cell import UnitCell
from cnf import CrystalNormalForm

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
delta = 20
xi = 1.1
# There are 642,000 unimodular mats
sample_freq = 2000
num_test_reqs = 100

import numpy as np

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

@helpers.parameterized_by_mp_structs
def test_bnf_is_unique(idx, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()

    original_vonorms = uc.superbasis.compute_vonorms()
    assert original_vonorms.is_obtuse(tol=1e-5)
    assert uc.is_obtuse(tol=1e-5)
    helpers.printif(f"Considering struct of class V{uc.voronoi_class}", verbose)
    
    bnfs = []
    lnfs = []
    for u in UNIMODULAR_MATRICES[::sample_freq]:
        other_cell = uc.apply_unimodular(u)
        if helpers.are_geo_matches(uc, other_cell, tol=1e-4):
            cnf = other_cell.to_cnf(xi, delta)
            bnfs.append(tuple(cnf.basis_normal_form.coord_list))
            lnfs.append(cnf.lattice_normal_form)

    assert len(lnfs) > num_test_reqs
    if len(set(lnfs)) == 1:
        assert len(set(bnfs)) == 1


@helpers.parameterized_by_mp_structs
def test_cnf_is_unique(idx, struct: Structure):
    # struct = struct.to_primitive()
    # rounded_abc = np.round(struct.lattice.abc, 1)
    # rounded_ang = np.round(struct.lattice.angles, 0)
    # lat_rounded = Lattice.from_parameters(*rounded_abc, *rounded_ang)
    # struct = Structure(lat_rounded, struct.species, struct.frac_coords)
    uc = UnitCell.from_pymatgen_structure(struct).reduce()
    original_vonorms = uc.superbasis.compute_vonorms()
    assert original_vonorms.is_obtuse(tol=1e-5)
    assert uc.is_obtuse(tol=1e-5)
    helpers.printif(f"Considering struct of class V{uc.voronoi_class}", verbose)
    
    cnf_map: dict[CrystalNormalForm, list] = {}
    all_cnfs: list[CrystalNormalForm] = []
    for u in UNIMODULAR_MATRICES[::sample_freq]:
        other_cell = uc.apply_unimodular(u)
        if helpers.are_geo_matches(uc, other_cell):
            cnf = other_cell.to_cnf(xi, delta)
            all_cnfs.append(cnf)
            if cnf.coords in cnf_map:
                cnf_map[cnf.coords].append(other_cell)
            else:
                cnf_map[cnf.coords] = [other_cell]
    
    assert len(all_cnfs) > 100
    cnfs = list(cnf_map.keys())

    # any_category_contains_95_percent = False
    # for cnf, cells in cnf_map.items():
    #     print(len(cells), 0.95 * len(all_cnfs))
    #     if len(cells) > 0.95 * len(all_cnfs):
    #         any_category_contains_95_percent = True
    # for c in cnfs:
    #     print(c)
    # assert len(all_cnfs) > 100
    # if len(cnfs) != 1:
    #     patho_sets_dir = pathlib.Path(__file__).parent / ".." / "data" / "patho_pairs"
    #     existing_dirs = list(patho_sets_dir.iterdir())
    #     current_set_name = f"mp_{idx}"
    #     existing_patho_dirs = [d.name for d in existing_dirs]
    #     if current_set_name not in existing_patho_dirs:
    #         os.makedirs(patho_sets_dir / current_set_name)
    #         idx = 0
    #         for cnf, cells in cnf_map.items():
    #             cells[0].to_cif(str(patho_sets_dir / current_set_name / f"{idx}.cif"))
    #             idx += 1

    assert len(set(cnfs)) == 1