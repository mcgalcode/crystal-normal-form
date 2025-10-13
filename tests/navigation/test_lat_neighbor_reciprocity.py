import pytest
import numpy as np
import helpers

from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.lattice.lnf_constructor import VonormCanonicalizer, LatticeNormalForm
from cnf.navigation.lattice_neighbors import LatticeStep, LatticeNeighborFinder
from pymatgen.core.structure import Structure

STRUCT_SAMPLE_FREQ = 1

@helpers.skip_if_fast
@pytest.mark.parametrize("struct", helpers.ALL_MP_STRUCTURES[::STRUCT_SAMPLE_FREQ])
def test_lnf_neighbor_reciprocity(struct: Structure):
    verbose = False
    xi = 1.5
    delta = 30
    helpers.printif("", verbose)
    # helpers.printif(f"Attempting struct at idx {idx}", verbose)
    constructor = CNFConstructor(xi, delta, False) 

    original_cnf = constructor.from_pymatgen_structure(struct).cnf
    original_lnf = original_cnf.lattice_normal_form
    original_v_class = original_lnf.vonorms.conorms.form.voronoi_class
    helpers.printif(f"Structure is of Voronoi Class V{original_v_class}", verbose)
    nb_set = LatticeNeighborFinder(original_cnf).find_lnf_neighbors()
    helpers.printif(f"Found {len(nb_set)} neighbors of the original structure...", verbose)
    for nb in nb_set.neighbors:
        nb_lnf = nb.point
        first_neighbor_v_class = nb_lnf.vonorms.conorms.form.voronoi_class
        second_neighbors = LatticeNeighborFinder(nb_lnf).find_lnf_neighbors()
        helpers.printif(f"Searching through {len(second_neighbors)} second-degree neighbors for original LNF to confirm reciprocity...", verbose)
        if original_lnf not in second_neighbors:
            helpers.printif(f"Neighbor is of Voronoi Class V{first_neighbor_v_class} (equal to original: {first_neighbor_v_class == original_v_class})", verbose)
            helpers.printif(f"No reciprocal relationship found!", verbose)

        assert original_lnf in second_neighbors
        helpers.printif(f"Original LNF was found!", verbose)
    
@helpers.skip_if_fast
@pytest.mark.parametrize("struct", helpers.ALL_MP_STRUCTURES[::STRUCT_SAMPLE_FREQ])
def test_lnf_neighbor_reciprocity_within_cnf_neighbors(struct: Structure):
    verbose = False
    xi = 1.5
    delta = 30
    helpers.printif("", verbose)
    # helpers.printif(f"Attempting struct at idx {idx}", verbose)
    constructor = CNFConstructor(xi, delta, False) 

    original_cnf = constructor.from_pymatgen_structure(struct).cnf

    neigb_set = LatticeNeighborFinder(original_cnf).find_cnf_neighbors()
    for n in neigb_set.neighbors:
        second_neighbor_set = LatticeNeighborFinder(n.point).find_cnf_neighbors()
        second_neighbor_lnfs = [sn.point.lattice_normal_form for sn in second_neighbor_set.neighbors]
        if original_cnf.lattice_normal_form not in second_neighbor_lnfs:
            print(f"No reciprocal relationship found!")
        assert original_cnf.lattice_normal_form in second_neighbor_lnfs

@helpers.skip_if_fast
@pytest.mark.parametrize("struct", helpers.ALL_MP_STRUCTURES[::10])
def test_neighbor_reciprocity_by_geometry(struct: Structure):
    verbose = False
    xi = 0.1
    delta = 20
    CUTOFF = 0.00000001
    struct = struct.to_primitive()
    # if len(struct) <= 2:
    
    print(f"Struct has {len(struct)} sites")
    cnf_constructor = CNFConstructor(xi, delta)
    original_cnf = cnf_constructor.from_pymatgen_structure(struct).cnf
    print(f"Original CNF: {original_cnf.coords}")

    neighbor_set = LatticeNeighborFinder(original_cnf).find_cnf_neighbors()
    print(f"Structure has {len(neighbor_set)} neighbors")
    reciprocal_nbs = [] 
    nonreciprocal_nbs = []

    for n in neighbor_set.neighbors:
        second_neighbors = LatticeNeighborFinder(n.point).find_cnf_neighbors()
        original_found = False
        for sn in second_neighbors.neighbors:
            neighb_cnf = sn.point
            pdd_dist = helpers.pdd_for_cnfs(original_cnf, neighb_cnf)
            if pdd_dist < CUTOFF:
                original_found = True
                assert neighb_cnf.lattice_normal_form == original_cnf.lattice_normal_form
        
        if original_found:
            reciprocal_nbs.append((n.point, pdd_dist))
            original_found = True
        else:
            reciprocal_nbs.append((n.point, None))

    print(f"Found {len(reciprocal_nbs)} GOOD neighbors")
    print(f"Found {len(nonreciprocal_nbs)} BAD neighbors")
    assert len(neighbor_set.neighbors) == len(reciprocal_nbs)

# @pytest.xfail("This just is not a true fact")
def test_neighbor_bnfs_yield_same_xtal_in_original_lattice():
    struct = helpers.ALL_MP_STRUCTURES[260]

    xi = 0.1
    delta = 10
    verbose = False
    constructor = CNFConstructor(xi, delta, verbose)
    original_cnf = constructor.from_pymatgen_structure(struct).cnf
    nf = LatticeNeighborFinder(original_cnf)
    cnf_neighb_set = nf.find_cnf_neighbors()
    tested_neighbs: list[LatticeStep] = []
    CUTOFF = 0.000000001
    dups = []
    for neighb in cnf_neighb_set.neighbors:
        constructor = CNFConstructor(xi, delta)
        
        new_neighb = neighb.point
        new_bnf = new_neighb.basis_normal_form
        zombie_cnf = constructor.from_discretized_vonorms_and_motif(original_cnf.lattice_normal_form.vonorms, new_bnf.to_discretized_motif()).cnf
        if zombie_cnf != original_cnf:
            assert zombie_cnf.lattice_normal_form == original_cnf.lattice_normal_form
            print(zombie_cnf.basis_normal_form.coord_list)
            print(original_cnf.basis_normal_form.coord_list)

@helpers.skip_if_fast
@helpers.parameterized_by_mp_structs
def test_cnf_neighbor_reciprocity(idx, struct: Structure):
    verbose = True
    xi = 0.001
    delta = 30

    helpers.printif("", verbose)
    helpers.printif(f"Attempting struct at idx {idx * STRUCT_SAMPLE_FREQ}", verbose)
    constructor = CNFConstructor(xi, delta, False) 

    struct = struct.to_primitive()
    
    print(f"Struct has {len(struct)} sites")

    original_cnf = constructor.from_pymatgen_structure(struct)
    original_xtal = original_cnf.reconstruct()
    # helpers.assert_identical_by_pdd_distance(struct, original_xtal, cutoff=0.1)

    print(f"Original CNF: {original_cnf.coords}")
    print(f"Original Voronoi: {original_cnf.voronoi_class}, {original_cnf.lattice_normal_form.vonorms.conorms.form}")
    CUTOFF = 0.015
    neighbor_set = LatticeNeighborFinder(original_cnf).find_cnf_neighbors()
    print(f"Structure has {len(neighbor_set)} neighbors")
    recipricol_nbs = [] 
    nonreciprocal_nbs = []
    geo_rec_neighbs = []

    for n in neighbor_set.neighbors:
        second_neighbors = LatticeNeighborFinder(n.point).find_cnf_neighbors()
        if original_cnf not in second_neighbors:
            print(f"No reciprocal relationship found!")
            nonreciprocal_nbs.append(n.point)
            num_geo_matches = 0
            for n2 in second_neighbors.neighbors:
                if n2.point.lattice_normal_form == original_cnf.lattice_normal_form:
                    dist = helpers.pdd(n2.point.reconstruct(), original_xtal)
                    if dist < CUTOFF and not helpers.are_cnfs_mirror_images(original_cnf, n2.point):
                        num_geo_matches += 1
                        print(f"NB V Class: {n.point.voronoi_class}, {original_cnf.lattice_normal_form.vonorms.conorms.form}")
                        print(n2.point.coords)
            if num_geo_matches > 0:
                geo_rec_neighbs.append(n)
            # assert num_geo_matches > 0
            helpers.printif(f"Found {num_geo_matches} geometrically identical second degree neighbs with same LNF!", verbose)
        else:
            recipricol_nbs.append(n)
    print(f"Found {len(recipricol_nbs)} GOOD neighbors")
    print(f"Found {len(nonreciprocal_nbs)} BAD neighbors")
    print(f"Found {len(geo_rec_neighbs)} GEO neighbors")
    assert len(recipricol_nbs) == len(neighbor_set.neighbors)

@helpers.parameterized_by_mp_struct_idxs([136])
def test_pathological_neighbor_recip_1(idx, struct):
    verbose = True
    xi = 1.5
    delta = 30

    helpers.printif("", verbose)
    helpers.printif(f"Attempting struct at idx {idx * STRUCT_SAMPLE_FREQ}", verbose)
    constructor = CNFConstructor(xi, delta, True) 

    struct = struct.to_primitive()
    
    print(f"Struct has {len(struct)} sites")

    original_cnf = constructor.from_pymatgen_structure(struct).cnf
    original_xtal = original_cnf.reconstruct()
    # helpers.assert_identical_by_pdd_distance(struct, original_xtal, cutoff=0.1)

    print(f"Original CNF: {original_cnf.coords}")
    print(f"Original Voronoi: {original_cnf.voronoi_class}")
    CUTOFF = 0.015
    neighbor_set = LatticeNeighborFinder(original_cnf).find_cnf_neighbors()
    print(f"Structure has {len(neighbor_set)} neighbors")

    nb_step_pairs = []

    found = False
    for n in neighbor_set.neighbors:
        if found:
            break
        second_neighbors = LatticeNeighborFinder(n.point).find_cnf_neighbors()
        if original_cnf not in second_neighbors:
            num_geo_matches = 0
            for n2 in second_neighbors.neighbors:
                if n2.point.lattice_normal_form == original_cnf.lattice_normal_form:
                    # print(np.array(n.coords) - np.array(original_cnf.coords))
                    # print()
                    dist = helpers.pdd(n2.point.reconstruct(), original_xtal)
                    if dist < CUTOFF:
                        num_geo_matches += 1
                        nb_step_pairs.append((n, second_neighbors.steps_for_neighbor(n2.point)))
                        found = True
                        break
            helpers.printif(f"Found {num_geo_matches} geometrically identical second degree neighbs with same LNF!", verbose)

    print(nb_step_pairs)
    nb = nb_step_pairs[0][0]
    step = nb_step_pairs[0][1][0]
    nb_f = LatticeNeighborFinder(nb.point, verbose_logging=True)
    nb_f.find_cnf_neighbor(step.step)

@helpers.parameterized_by_mp_structs
def test_second_neighbors_obey_reciprocity(idx, struct):
    verbose = False
    xi = 2.0
    delta = 30

    helpers.printif("", verbose)
    helpers.printif(f"Attempting struct at idx {idx * STRUCT_SAMPLE_FREQ}", verbose)
    constructor = CNFConstructor(xi, delta, False) 

    struct = struct.to_primitive()
    
    helpers.printif(f"Struct has {len(struct)} sites", verbose)

    original_cnf = constructor.from_pymatgen_structure(struct)
    original_xtal = original_cnf.reconstruct()
    # helpers.assert_identical_by_pdd_distance(struct, original_xtal, cutoff=0.1)

    CUTOFF = 0.015
    nf = LatticeNeighborFinder(original_cnf)

    for step in nf.possible_steps():
        nb = nf.find_cnf_neighbor(step)
        if nb is not None:
            break

    cnf_2 = nb.result
    helpers.printif(f"Original Voronoi: {cnf_2.voronoi_class}", verbose)
    helpers.printif(f"Original CNF: {cnf_2.coords}", verbose)
    nf2 = LatticeNeighborFinder(cnf_2)
    neighbor_set = nf2.find_cnf_neighbors()
    helpers.printif(f"Structure has {len(neighbor_set)} neighbors", verbose)
    recipricol_nbs = [] 
    nonreciprocal_nbs = []
    geo_rec_neighbs = []

    for n in neighbor_set.neighbors:
        second_neighbors = LatticeNeighborFinder(n.point).find_cnf_neighbors()
        if cnf_2 not in second_neighbors:
            helpers.printif(f"No reciprocal relationship found!", verbose)
            nonreciprocal_nbs.append(n.point)
            num_geo_matches = 0
            for n2 in second_neighbors.neighbors:
                if n2.point.lattice_normal_form == cnf_2.lattice_normal_form:
                    # helpers.printif(n2.point, verbose)
                    dist = helpers.pdd(n2.point.reconstruct(), original_xtal)
                    if dist < CUTOFF:
                        assert not helpers.are_cnfs_mirror_images(cnf_2, n2.point)
                        num_geo_matches += 1
                        helpers.printif(f"NB V Class: {n.point.voronoi_class}, {cnf_2.lattice_normal_form.vonorms.conorms.form}", verbose)
                        helpers.printif(n2.point.coords, verbose)
            if num_geo_matches > 0:
                geo_rec_neighbs.append(n)
            # assert num_geo_matches > 0
            helpers.printif(f"Found {num_geo_matches} geometrically identical second degree neighbs with same LNF!", verbose)
        else:
            recipricol_nbs.append(n)
    helpers.printif(f"Found {len(recipricol_nbs)} GOOD neighbors", verbose)
    helpers.printif(f"Found {len(nonreciprocal_nbs)} BAD neighbors", verbose)
    helpers.printif(f"Found {len(geo_rec_neighbs)} GEO neighbors", verbose)
    assert len(recipricol_nbs) == len(neighbor_set.neighbors)
