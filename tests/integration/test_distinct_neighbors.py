import pytest
import numpy as np
import helpers

from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.lattice.lnf_constructor import VonormCanonicalizer, LatticeNormalForm
from cnf.navigation.lattice_neighbor_finder import LatticeStep, LatticeNeighborFinder, LatticeStepResult
from pymatgen.core.structure import Structure
from cnf.unit_cell import UnitCell


@helpers.parameterized_by_mp_structs
def test_neighbors_are_geometrically_distinct(idx, struct: Structure):
    # if not helpers.does_struct_have_centrosymmetric_symm(struct) and not helpers.is_struct_chiral(struct):
    #     return
    # if len(struct) >= 4:
    #     return
    verbose = False
    xi = 1.5
    delta = 20
    constructor = CNFConstructor(xi, delta, False)
    original_cnf = constructor.from_pymatgen_structure(struct).cnf
    nf = LatticeNeighborFinder(original_cnf, verbose_logging=True)
    cnf_neighb_set = nf.find_cnf_neighbors()
    tested_neighbs: list[CrystalNormalForm] = []
    dups = []

    clusters = {}

    chiral_neighbs = 0
    helpers.printif(f"Original structure is Voronoi: {original_cnf.lattice_normal_form.vonorms.conorms.form.voronoi_class}", verbose)
    for neighb in cnf_neighb_set.neighbors:
        # if not helpers.is_struct_chiral(neighb.point.reconstruct()):
        #     continue
        steps = cnf_neighb_set.steps_for_neighbor(neighb.point)
        current_cnf = neighb.point
        is_duplicate = False
        dup = None

        for existing_cnf in clusters:
            # if existing_cnf.lattice_normal_form != current_cnf.lattice_normal_form:
            #     continue
            pdd_dist = helpers.pdd_for_cnfs(current_cnf, existing_cnf)
            match, reason = helpers.are_cnfs_geo_matches(current_cnf, existing_cnf, tol=1e-12)
            if match:
                is_duplicate = True
                verbose = False
                print()
                print(existing_cnf.lattice_normal_form)
                print(current_cnf.lattice_normal_form)
                print()
                assert existing_cnf.lattice_normal_form == current_cnf.lattice_normal_form
                helpers.printif(existing_cnf.basis_normal_form.coord_list, verbose)
                helpers.printif(current_cnf.basis_normal_form.coord_list, verbose)
                helpers.printif(f"PDD Distance: {pdd_dist}", verbose)
                helpers.printif(f"Neighbor 1 is Voronoi: {existing_cnf.voronoi_class}", verbose)
                helpers.printif(f"Neighbor 2 is Voronoi: {current_cnf.voronoi_class}", verbose)
                helpers.printif("Steps leading to new point:", verbose)
                for step in steps:
                    if verbose:
                        step.step.print_details()
                    helpers.printif(step.construction_result.cnf.basis_normal_form, verbose)
                
                helpers.printif("", verbose)
                helpers.printif(f"Steps leading to old point: ", verbose)
                if verbose:
                    for step in cnf_neighb_set.steps_for_neighbor(existing_cnf):
                        step.step.print_details()
                helpers.printif(f"This neighbor was reached by {len(steps)} steps", verbose)
                clusters[existing_cnf].append(current_cnf)
                dups.append((current_cnf, dup, pdd_dist))
                helpers.printif("", verbose)

        if not is_duplicate:
            clusters[current_cnf] = [current_cnf]
            tested_neighbs.append(neighb)

    
    print()
    if not len(clusters) == len(cnf_neighb_set):
        cidx = 0
        for _, identical_neighbs in clusters.items():
            if len(identical_neighbs) > 1:
                print(f"{cidx} Cluster")
                for nb in identical_neighbs:
                    print(nb.lattice_normal_form)
                helpers.save_cnfs_to_dir(f"geo_pairs_neighbs/mp_{idx}/cluster_{cidx}", identical_neighbs)
                helpers.save_cifs_to_dir(f"geo_pairs_neighbs_cifs/mp_{idx}/cluster_{cidx}", [c.reconstruct() for c in identical_neighbs])
            cidx += 1
    
    assert len(clusters) == len(cnf_neighb_set)
    # assert len(clusters) == chiral_neighbs