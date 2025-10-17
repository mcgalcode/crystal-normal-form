import pytest
import numpy as np
import helpers

from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.lattice.lnf_constructor import VonormCanonicalizer, LatticeNormalForm
from cnf.navigation.basis_neighbor_finder import BasisStepResult, BasisNeighborFinder
from pymatgen.core.structure import Structure
from cnf.unit_cell import UnitCell


@helpers.parameterized_by_structs_with_num_sites_less_than(10)
def test_neighbors_are_geometrically_distinct(idx, struct: Structure):    
    verbose = False
    xi = 1.5
    delta = 20
    constructor = CNFConstructor(xi, delta, False)
    original_cnf = constructor.from_pymatgen_structure(struct).cnf
    nf = BasisNeighborFinder(original_cnf)
    neighb_set = nf.find_basis_neighbors()
    tested_neighbs: list[CrystalNormalForm] = []
    dups = []

    clusters = {}

    helpers.printif(f"Original structure is Voronoi: {original_cnf.lattice_normal_form.vonorms.conorms.form.voronoi_class}", verbose)
    for neighb in neighb_set.neighbors:
        steps = neighb_set.steps_for_neighbor(neighb.point)
        current_cnf = neighb.point
        is_duplicate = False
        dup = None

        for existing_cnf in clusters:
            pdd_dist = helpers.pdd_for_cnfs(current_cnf, existing_cnf)
            match, reason = helpers.are_cnfs_geo_matches(current_cnf, existing_cnf, tol=1e-12)
            if match:
                is_duplicate = True
                verbose = False
                print(existing_cnf.basis_normal_form.coord_list)
                print(current_cnf.basis_normal_form.coord_list)
                helpers.printif(f"This neighbor was reached by {len(steps)} steps", verbose)
                clusters[existing_cnf].append(current_cnf)
                dups.append((current_cnf, dup, pdd_dist))
                helpers.printif("", verbose)

        if not is_duplicate:
            clusters[current_cnf] = [current_cnf]
            tested_neighbs.append(neighb)

    
    print()
    if not len(clusters) == len(neighb_set):
        cidx = 0
        for _, identical_neighbs in clusters.items():
            if len(identical_neighbs) > 1:
                print(f"{cidx} Cluster")
                for nb in identical_neighbs:
                    print(nb.basis_normal_form)
                helpers.save_cnfs_to_dir(f"geo_pairs_basis_neighbs/mp_{idx}/cluster_{cidx}", identical_neighbs)
                helpers.save_cifs_to_dir(f"geo_pairs_basis_neighbs_cifs/mp_{idx}/cluster_{cidx}", [c.reconstruct() for c in identical_neighbs])
            cidx += 1
    
    assert len(clusters) == len(neighb_set)
    # assert len(clusters) == chiral_neighbs