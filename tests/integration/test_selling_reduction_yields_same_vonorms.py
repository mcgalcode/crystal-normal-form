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
from cnf.lattice.selling import VonormListSellingReducer
from cnf.lattice.voronoi.vonorm_list import VonormList
from cnf.lattice.lnf_constructor import VonormSorter, VonormCanonicalizer
from cnf.lattice.rounding import DiscretizedVonormComputer
from cnf.linalg.matrix_tuple import MatrixTuple
from cnf.linalg.unimodular import combine_unimodular_mats_np
import pathlib
from itertools import product
from cnf.linalg.unimodular import UNIMODULAR_MATRICES

verbose = False
delta = 100
xi = 0.001
# There are 642,000 unimodular mats
sample_freq = 20000
num_test_reqs = 100

import numpy as np


@helpers.parameterized_by_mp_structs
def test_selling_reduction_vonorm_set_is_unique(idx, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()
    helpers.printif(f"Considering struct of class V{uc.voronoi_class}", verbose)
    
    cnf_map: dict[CrystalNormalForm, list] = {}
    all_cnfs: list[CrystalNormalForm] = []
    r = VonormListSellingReducer()
    original = uc.vonorms
    for u in UNIMODULAR_MATRICES[::sample_freq]:
        other_cell = uc.apply_unimodular(u)
        match, reason = helpers.are_unit_cells_geo_matches(uc, other_cell, tol=1e-6)
        if match:
            reduced_other = r.reduce(other_cell.vonorms).reduced_object
            reduced_other = reduced_other
            assert original.has_same_members(reduced_other, tol=1e-3)


@helpers.parameterized_by_mp_structs
def test_reduction_and_sorting_yields_same_ordered_list(idx, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()
    helpers.printif(f"Considering struct of class V{uc.voronoi_class}", verbose)
    
    cnf_map: dict[CrystalNormalForm, list] = {}
    all_cnfs: list[CrystalNormalForm] = []
    r = VonormListSellingReducer()
    original = uc.vonorms
    sorter = VonormSorter()
    original_canonicalzed, original_transforms = sorter.get_canonicalized_vonorms(original)
    for u in UNIMODULAR_MATRICES[::sample_freq]:
        other_cell = uc.apply_unimodular(u)
        match, reason = helpers.are_unit_cells_geo_matches(uc, other_cell, tol=1e-6)
        if match:
            reduced_other = r.reduce(other_cell.vonorms).reduced_object
            reduced_other: VonormList = reduced_other
            assert reduced_other.conorms.form.voronoi_class == original_canonicalzed.conorms.form.voronoi_class
            assert original.has_same_members(reduced_other, tol=1e-5)
            other_canonical, other_transforms = sorter.get_canonicalized_vonorms(reduced_other)
            assert len(original_transforms) == len(other_transforms)
            assert other_canonical == original_canonicalzed

@helpers.parameterized_by_mp_structs
def test_reduction_sorting_and_discretizing_yields_same_ordered_list(idx, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()
    helpers.printif(f"Considering struct of class V{uc.voronoi_class}", verbose)
    
    cnf_map: dict[CrystalNormalForm, list] = {}
    all_cnfs: list[CrystalNormalForm] = []
    r = VonormListSellingReducer()
    original = uc.vonorms
    sorter = VonormSorter()
    original_canonicalzed, original_transforms = sorter.get_canonicalized_vonorms(original)
    dvc = DiscretizedVonormComputer(1.5)
    original_discrete = dvc.find_closest_valid_vonorms(original_canonicalzed)
    for u in UNIMODULAR_MATRICES[::sample_freq]:
        other_cell = uc.apply_unimodular(u)
        match, reason = helpers.are_unit_cells_geo_matches(uc, other_cell, tol=1e-6)
        if match:
            reduced_other = r.reduce(other_cell.vonorms).reduced_object
            reduced_other: VonormList = reduced_other.set_tol(1e-4)
            assert reduced_other.conorms.form.voronoi_class == original_canonicalzed.conorms.form.voronoi_class
            assert original.has_same_members(reduced_other, tol=1e-5)
            other_canonical, other_transforms = sorter.get_canonicalized_vonorms(reduced_other)
            assert len(original_transforms) == len(other_transforms)
            assert other_canonical == original_canonicalzed
            other_discrete = dvc.find_closest_valid_vonorms(other_canonical)
            assert other_discrete == original_discrete
            assert set(other_discrete.stabilizer_matrices()) == set(original_discrete.stabilizer_matrices())