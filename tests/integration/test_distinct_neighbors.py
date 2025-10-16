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
    # if len(struct) >= 4:
    #     return
    verbose = True
    xi = 1.0
    delta = 20
    constructor = CNFConstructor(xi, delta, False)
    original_cnf = constructor.from_pymatgen_structure(struct).cnf
    nf = LatticeNeighborFinder(original_cnf)
    cnf_neighb_set = nf.find_cnf_neighbors()
    tested_neighbs: list[CrystalNormalForm] = []
    CUTOFF = 0.00001
    dups = []

    helpers.printif(f"Original structure is Voronoi: {original_cnf.lattice_normal_form.vonorms.conorms.form.voronoi_class}", verbose)
    for neighb in cnf_neighb_set.neighbors:
        steps = cnf_neighb_set.steps_for_neighbor(neighb.point)
        new_neighb = neighb.point
        is_duplicate = False
        dup = None

        for old_neighb_res in tested_neighbs:
            old_neighb = old_neighb_res.point
            pdd_dist = helpers.pdd_for_cnfs(new_neighb, old_neighb)
            if helpers.are_cnfs_geo_matches(new_neighb, old_neighb, tol=1e-4):
                verbose = True
                helpers.save_cnfs_to_dir(f"geo_pairs_neighbs/mp_{idx}", [new_neighb, old_neighb])

                # recovered1 = constructor.from_pymatgen_structure(UnitCell.from_cnf(new_neighb).to_pymatgen_structure())
                # recovered2 = constructor.from_pymatgen_structure(UnitCell.from_cnf(old_neighb).to_pymatgen_structure())
                # assert recovered1.cnf == recovered2.cnf
                # UnitCell.from_cnf(old_neighb).to_pymatgen_structure().to_file("nb1.cif")
                # UnitCell.from_cnf(new_neighb).to_pymatgen_structure().to_file("nb2.cif")
                # return
                assert old_neighb.lattice_normal_form == new_neighb.lattice_normal_form
                helpers.printif(old_neighb.basis_normal_form.coord_list, verbose)
                helpers.printif(new_neighb.basis_normal_form.coord_list, verbose)
                helpers.printif(f"PDD Distance: {pdd_dist}", verbose)
                # helpers.printif(old_neighb.basis_normal_form.elements, verbose)
                helpers.printif(f"Neighbor 1 is Voronoi: {old_neighb.voronoi_class}", verbose)
                helpers.printif(f"Neighbor 2 is Voronoi: {new_neighb.voronoi_class}", verbose)
                
                # old_motif = old_neighb.basis_normal_form.to_discretized_motif()
                # new_motif = new_neighb.basis_normal_form.to_discretized_motif()
                # transform = old_motif.coord_matrix @ np.linalg.pinv(new_motif.coord_matrix)
                # helpers.printif(f"Related by:", verbose)
                # helpers.printif(transform, verbose)
                # helpers.printif(new_neighb.basis_normal_form.elements, verbose)
                helpers.printif("Steps leading to new point:", verbose)
                for step in steps:
                    if verbose:
                        step.step.print_details()
                    helpers.printif(step.construction_result.cnf.basis_normal_form, verbose)
                    assert step.step.prereq_perm.matrix.determinant() == 1
                
                helpers.printif("", verbose)
                helpers.printif(f"Steps leading to old point: ", verbose)
                if verbose:
                    for step in cnf_neighb_set.steps_for_neighbor(old_neighb):
                        step.step.print_details()
                is_duplicate = True
                dup = old_neighb

        if is_duplicate:
            helpers.printif(f"This neighbor was reached by {len(steps)} steps", verbose)
            dups.append((new_neighb, dup, pdd_dist))
            helpers.printif("", verbose)
        else:
            tested_neighbs.append(neighb)
    
    assert len(tested_neighbs) == len(cnf_neighb_set)