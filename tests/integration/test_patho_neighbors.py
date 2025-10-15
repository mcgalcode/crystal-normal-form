import helpers
import pytest
import numpy as np

from cnf import CrystalNormalForm

from cnf.navigation import LatticeNeighborFinder

def assert_reciprocal_geo_neighbors(cnf1, cnf2, nbs1, nbs2):
    geo_found_2_in_1 = False
    for nb in nbs1.neighbors:
        if helpers.are_cnfs_geo_matches(nb.point, cnf2):
            geo_found_2_in_1 = True
    
    geo_found_1_in_2 = False
    for nb in nbs2.neighbors:
        if helpers.are_cnfs_geo_matches(nb.point, cnf1):
            geo_found_1_in_2 = True

    assert geo_found_1_in_2
    assert geo_found_2_in_1

def assert_reciprocal_lnf_neighbors(cnf1, cnf2, nbs1, nbs2):
    nbs1_lnfs = [nb.point.lattice_normal_form for nb in nbs1.neighbors]
    nbs2_lnfs = [nb.point.lattice_normal_form for nb in nbs2.neighbors]
    
    assert cnf1.lattice_normal_form in nbs2_lnfs
    assert cnf2.lattice_normal_form in nbs1_lnfs


def test_are_neighbors_a_pair():
    cnfs: list[CrystalNormalForm] = helpers.data.load_pathological_neighbors("mp_13_nb_15")
    
    cnf1 = cnfs[1]
    cnf2 = cnfs[0]
    print()
    print(f"PDD dist: {helpers.pdd_for_cnfs(cnf1, cnf2)}")
    print()
    print(f"CNF1 LNF: {cnf1.lattice_normal_form.coords}")
    print(f"CNF1 BNF: {cnf1.basis_normal_form.coord_list}")
    print(f"CNF1 Voronoi Class: {cnf1.voronoi_class}")
    print()
    print(f"CNF2 LNF: {cnf2.lattice_normal_form.coords}")
    print(f"CNF2 BNF: {cnf2.basis_normal_form.coord_list}")
    print(f"CNF2 Voronoi Class: {cnf2.voronoi_class}")
    lf1 = LatticeNeighborFinder(cnf1)
    lf2 = LatticeNeighborFinder(cnf2)


    nbs1 = lf1.find_cnf_neighbors()
    nbs2 = lf2.find_cnf_neighbors()
    for n in nbs2.neighbors:
        if n.point.lattice_normal_form == cnf1.lattice_normal_form:
            steps = n.step_results
            for s in steps:
                s.print_details()

    assert_reciprocal_lnf_neighbors(cnf1, cnf2, nbs1, nbs2)
    assert_reciprocal_geo_neighbors(cnf1, cnf2, nbs1, nbs2)

    assert cnf1 in nbs2
    assert cnf2 in nbs1
    