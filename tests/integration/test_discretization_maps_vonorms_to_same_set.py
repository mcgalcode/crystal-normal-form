import pytest
import helpers
import tqdm
import os

from pymatgen.core.structure import Structure
from cnf.unit_cell import UnitCell
from cnf.lattice.rounding import DiscretizedVonormComputer
from cnf.lattice.voronoi import VonormList
from cnf.lattice.lnf_constructor import VonormCanonicalizer

import pathlib

from cnf.linalg.unimodular import UNIMODULAR_MATRICES

verbose = False
delta = 20
xi = 1.1

@helpers.parameterized_by_mp_structs
def test_rounding_vonorms_maps_to_same_vonorm_set(idx, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()

    original_vonorms = uc.superbasis.compute_vonorms()
    
    disc_vonorm_sets = []

    dvc = DiscretizedVonormComputer(xi)
    print()
    helpers.printif(f"Original: {dvc.uncorrected_discretized_vonorms(original_vonorms)}", verbose)

    for u in UNIMODULAR_MATRICES[::10]:
        other_cell = uc.apply_unimodular(u).reduce()
        pdd = helpers.pdd(struct, other_cell.to_pymatgen_structure())
        if pdd < 1e-5:
            rounded = dvc.uncorrected_discretized_vonorms(other_cell.vonorms)
            helpers.printif(f"Original: {dvc.uncorrected_discretized_vonorms(other_cell.vonorms)}", verbose)

            vals = sorted(rounded.vonorms)
            disc_vonorm_sets.append(tuple(vals))
    assert len(disc_vonorm_sets) > 100
    assert len(set(disc_vonorm_sets)) == 1

@helpers.parameterized_by_mp_structs
def test_canonicalizing_leads_to_same_vonorm_list(idx, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()

    original_vonorms = uc.superbasis.compute_vonorms()
    
    can_vonorm_sets = []
    can_conorm_sets = []
    can = VonormCanonicalizer()
    dvc = DiscretizedVonormComputer(xi)
    print()
    helpers.printif(f"Original: {dvc.uncorrected_discretized_vonorms(original_vonorms)}", verbose)

    for u in UNIMODULAR_MATRICES[::10]:
        other_cell = uc.apply_unimodular(u).reduce()
        pdd = helpers.pdd(struct, other_cell.to_pymatgen_structure())
        if pdd < 1e-5:
            canonicalized = can.get_canonicalized_vonorms(other_cell.vonorms, skip_reduction=True, coform_tolerance=0.01).canonical_vonorms
            can_vonorm_sets.append(tuple([float(v.round(2)) for v in canonicalized.vonorms]))
            can_conorm_sets.append(tuple([float(c.round(2)) for c in canonicalized.conorms.conorms]))
    assert len(can_vonorm_sets) > 100
    for v in set(can_vonorm_sets):
        print(v)
    print()
    for v in set(can_conorm_sets):
        print(v)
    assert len(set(can_vonorm_sets)) == 1

@helpers.parameterized_by_mp_structs
def test_canonicalizing_then_rounding_vonorms_maps_to_same_vonorm_list(idx, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()

    original_vonorms = uc.superbasis.compute_vonorms()
    
    disc_vonorm_sets = []

    can = VonormCanonicalizer()
    dvc = DiscretizedVonormComputer(xi)
    print()
    helpers.printif(f"Original: {dvc.uncorrected_discretized_vonorms(original_vonorms)}", verbose)

    for u in UNIMODULAR_MATRICES[::10]:
        other_cell = uc.apply_unimodular(u).reduce()
        pdd = helpers.pdd(struct, other_cell.to_pymatgen_structure())
        if pdd < 1e-5:
            canonicalized = can.get_canonicalized_vonorms(other_cell.vonorms, skip_reduction=True, coform_tolerance=0.01).canonical_vonorms
            rounded = dvc.uncorrected_discretized_vonorms(canonicalized)
            helpers.printif(f"Original: {dvc.uncorrected_discretized_vonorms(other_cell.vonorms)}", verbose)
            disc_vonorm_sets.append(tuple(rounded.vonorms))
    assert len(disc_vonorm_sets) > 100
    assert len(set(disc_vonorm_sets)) == 1

@helpers.parameterized_by_mp_structs
def test_canonicalizing_then_discretizing_vonorms_maps_to_same_vonorm_list(idx, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()

    original_vonorms = uc.superbasis.compute_vonorms()
    
    disc_vonorm_sets = []
    can_conorm_sets = []
    can = VonormCanonicalizer()
    dvc = DiscretizedVonormComputer(xi)
    print()
    helpers.printif(f"Original: {dvc.uncorrected_discretized_vonorms(original_vonorms)}", verbose)
    rounded_v_sets = []

    dv_map = {}
    rv_map = {}
    
    for u in UNIMODULAR_MATRICES[::10]:
        other_cell = uc.apply_unimodular(u).reduce()
        pdd = helpers.pdd(struct, other_cell.to_pymatgen_structure())
        if pdd < 1e-5:
            canonicalized = can.get_canonicalized_vonorms(other_cell.vonorms, skip_reduction=True, coform_tolerance=0.01).canonical_vonorms
            rounded = dvc.uncorrected_discretized_vonorms(canonicalized)
            rounded_v_sets.append(rounded.tuple)
            discretized = dvc.find_closest_valid_vonorms(canonicalized)
            disc_vonorm_sets.append(tuple([v for v in discretized]))
            can_conorm_sets.append(tuple([c for c in discretized.conorms]))

            if discretized.tuple in dv_map:
                dv_map[discretized.tuple].append((u, rounded.tuple, canonicalized.tuple))
            else:
                dv_map[discretized.tuple] = [(u, rounded.tuple, canonicalized.tuple)]
            
            if rounded.tuple in rv_map:
                rv_map[rounded.tuple].append(u)
            else:
                rv_map[rounded.tuple] = [u]
                
    
    assert len(set(rounded_v_sets)) == 1
    assert len(disc_vonorm_sets) > 10
    # for cv in dv_map:
    #     print()
    #     print(cv, len(dv_map[cv]))
    #     print(dv_map[cv][0][1])
    #     print(dv_map[cv][0][2])
    # print()
    # for v in set(can_conorm_sets):
    #     print(v)
    assert len(set(disc_vonorm_sets)) == 1


# The hypothesis is that this approach resolves the problem
@helpers.parameterized_by_mp_structs
def test_canonicalizing_then_discretizing_then_canonicalizing_makes_vonorm_list_unique(idx, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()    
    canonicalized = []

    c = VonormCanonicalizer(reduction_tolerance=1e-5)
    dvc = DiscretizedVonormComputer(xi)

    for u in UNIMODULAR_MATRICES[::10]:
        other_cell = uc.apply_unimodular(u).reduce()
        pdd = helpers.pdd(struct, other_cell.to_pymatgen_structure())
        if pdd < 1e-5:
            vonorms = c.get_canonicalized_vonorms(other_cell.vonorms).canonical_vonorms
            vonorms = dvc.find_closest_valid_vonorms(vonorms)
            vonorms = c.get_canonicalized_vonorms(vonorms)

            canonicalized.append(tuple(vonorms.canonical_vonorms))
    assert len(canonicalized) > 100
    if not len(set(canonicalized)) == 1:
        print(list(set(canonicalized)))
    assert len(set(canonicalized)) == 1

@helpers.parameterized_by_mp_structs
def test_discretized_vonorms_have_same_permissible_perms(idx, struct: Structure):
    uc = UnitCell.from_pymatgen_structure(struct).reduce()
    
    perm_sets = []
    c = VonormCanonicalizer(reduction_tolerance=1e-5)

    dvc = DiscretizedVonormComputer(xi)

    for u in UNIMODULAR_MATRICES[::10]:
        other_cell = uc.apply_unimodular(u).reduce()
        pdd = helpers.pdd(struct, other_cell.to_pymatgen_structure())
        if pdd < 1e-5:
            vonorms = c.get_canonicalized_vonorms(other_cell.vonorms, coform_tolerance=0.01).canonical_vonorms
            rounded = dvc.find_closest_valid_vonorms(vonorms)
            vals = sorted([p.vonorm_permutation.perm for p in rounded.conorms.permissible_permutations])
            perm_sets.append(tuple(vals))
    assert len(perm_sets) > 100
    assert len(set(perm_sets)) == 1
