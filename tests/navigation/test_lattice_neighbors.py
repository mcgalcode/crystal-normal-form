import pytest
import numpy as np
import helpers

from cnf import CrystalNormalForm
from cnf.lattice.lnf_constructor import VonormCanonicalizer
from cnf.navigation.lattice_neighbors import LatticeStep, LatticeNeighborFinder

def test_breaks_if_vec_has_non_one_value():
    vec = [0,0,0,2,0,0,0]
    with pytest.raises(ValueError) as excep:
        LatticeStep(vec)
    
    assert "invalid element != 1" in excep.value.__repr__()

def test_can_find_all_lattice_steps():
    all_steps = LatticeStep.all_step_vecs()
    assert len(all_steps) == 42
    assert len(set([s.tuple for s in all_steps])) == 42

def test_expanded_chi_set_for_structure(mp_structures):
    for struct in mp_structures:
        cnf = CrystalNormalForm.from_pymatgen_structure(struct)
        chi = LatticeStep.step_vecs_for_vonorm_list(cnf.lattice_normal_form.vonorms)
        assert len(chi) >= 42


def test_lattice_neighbor_lnfs_make_sense(mp_structures):
    struct = mp_structures[0]
    original_cnf = CrystalNormalForm.from_pymatgen_structure(struct)

    neighbors = LatticeNeighborFinder().find_lnf_neighbors(original_cnf.lattice_normal_form)

    for n in neighbors:
        diff = np.array(sorted(n.coords)) - np.array(sorted(original_cnf.lattice_normal_form.coords))
        assert np.sum(np.abs(diff)) == 2
        assert np.max(np.abs(diff)) == 1

def test_lattice_second_neighbor_lnfs_make_sense(mp_structures):
    struct = mp_structures[0]
    for struct in mp_structures[::50]:
        original_cnf = CrystalNormalForm.from_pymatgen_structure(struct)

        neighbors = LatticeNeighborFinder().find_lnf_neighbors(original_cnf.lattice_normal_form)

        for n in neighbors:
            second_neighbs = LatticeNeighborFinder().find_lnf_neighbors(n)
            for sn in second_neighbs:
                diff = np.array(sorted(sn.coords)) - np.array(sorted(original_cnf.lattice_normal_form.coords))
                assert np.sum(np.abs(diff)) <= 4
                assert np.max(np.abs(diff)) <= 2
        print(f"Structure was valid!")

def test_lattice_neighbor_cnfs_make_sense(mp_structures):
    for struct in mp_structures:
        original_cnf = CrystalNormalForm.from_pymatgen_structure(struct)

        neighbors = LatticeNeighborFinder().find_cnf_neighbors(original_cnf)

        for n in neighbors[:200]:
            diff = np.array(sorted(n.lattice_normal_form.coords)) - np.array(sorted(original_cnf.lattice_normal_form.coords))
            assert np.sum(np.abs(diff)) <= 2
            assert np.max(np.abs(diff)) <= 1

@pytest.mark.skip
def test_lnf_neighbor_reciprocity_pathological_case_1(mp_structures):
    verbose = True
    struct = mp_structures[50]

    original_lnf = CrystalNormalForm.from_pymatgen_structure(struct).lattice_normal_form
    helpers.printif(f"Original LNF vonorms: {original_lnf.vonorms}", verbose)
    canonicalizer = VonormCanonicalizer()
    c_result = canonicalizer.get_canonicalized_vonorms(original_lnf.vonorms, skip_reduction=True)
    canonicalized_vonorms = c_result.canonical_vonorms        
    helpers.printif(f"Re-Canonicalized original LNF vonorms: {canonicalized_vonorms}", verbose)
    original_v_class = original_lnf.vonorms.conorms.form.voronoi_class

    pathological_step = LatticeStep((0, 0, 0, 0, 1, -1, 0))
    helpers.printif(f"Structure is of Voronoi Class V{original_v_class}", verbose)
    helpers.printif(f"Considering pathological step: {pathological_step.tuple}", verbose)
    nf = LatticeNeighborFinder(verbose_logging=True)
    neighbor = nf.get_vonorm_neighbor(original_lnf.vonorms, pathological_step)
    assert neighbor.vonorms.is_obtuse()
    assert neighbor.vonorms.is_superbasis()
    assert neighbor.vonorms.to_superbasis().is_obtuse(tol=1e-5)
    helpers.printif("", verbose)
    neighbors_of_neighbor = nf.find_lnf_neighbors(neighbor)

    neighbor_lnfs = [nn[0] for nn in neighbors_of_neighbor]
    for nn in neighbors_of_neighbor:
        assert nn[0].vonorms.is_obtuse()
        assert nn[0].vonorms.is_superbasis()
    assert original_lnf in neighbor_lnfs

def test_lnf_neighbor_reciprocity(mp_structures):
    verbose = False
    for idx, struct in enumerate(mp_structures[::20]):
        helpers.printif("", verbose)
        helpers.printif(f"Attempting struct at idx {idx}", verbose)
        # struct = mp_structures[1]
        original_lnf = CrystalNormalForm.from_pymatgen_structure(struct).lattice_normal_form
        original_v_class = original_lnf.vonorms.conorms.form.voronoi_class
        helpers.printif(f"Structure is of Voronoi Class V{original_v_class}", verbose)
        neighbors = LatticeNeighborFinder().find_lnf_neighbors(original_lnf)
        helpers.printif(f"Found {len(neighbors)} neighbors of the original structure...", verbose)
        for neighbor_and_step in neighbors:
            # n, step, vonorm_perm = neighbor_and_step
            n = neighbor_and_step
            first_neighbor_v_class = n.vonorms.conorms.form.voronoi_class
            second_neighbors = [ns for ns in LatticeNeighborFinder().find_lnf_neighbors(n)]
            helpers.printif(f"Searching through {len(second_neighbors)} second-degree neighbors for original LNF to confirm reciprocity...", verbose)
            if original_lnf not in second_neighbors:
                helpers.printif(f"Neighbor is of Voronoi Class V{first_neighbor_v_class} (equal to original: {first_neighbor_v_class == original_v_class})", verbose)
                helpers.printif(f"No reciprocal relationship found!", verbose)

            assert original_lnf in second_neighbors
            helpers.printif(f"Original LNF was found!", verbose)
    

def test_lnf_neighbor_reciprocity_within_cnf_neighbors(mp_structures):
    verbose = False

    for idx, struct in enumerate(mp_structures[::50]):
        original_cnf = CrystalNormalForm.from_pymatgen_structure(struct)

        neighbors = LatticeNeighborFinder().find_cnf_neighbors(original_cnf)
        for n in neighbors:
            second_neighbors = LatticeNeighborFinder().find_cnf_neighbors(n)
            second_neighbor_lnfs = [sn.lattice_normal_form for sn in second_neighbors]
            if original_cnf.lattice_normal_form not in second_neighbor_lnfs:
                print(f"No reciprocal relationship found!")
            assert original_cnf.lattice_normal_form in second_neighbor_lnfs
    print("Success!")



def test_cnf_neighbor_reciprocity(mp_structures):
    verbose = False

    step = 50
    # for idx, struct in enumerate(mp_structures[::step]):

        # print(f"Trying struct at idx {idx * step}")
    struct = mp_structures[150]
    # print(f"Struct has {len(struct)} sites")
    original_cnf = CrystalNormalForm.from_pymatgen_structure(struct)
    print(f"Original CNF: {original_cnf.coords}")
    neighbors = LatticeNeighborFinder().find_cnf_neighbors(original_cnf)
    for n in neighbors:
        second_neighbors = LatticeNeighborFinder().find_cnf_neighbors(n)
        if original_cnf not in second_neighbors:
            print(f"No reciprocal relationship found!")
            for n in second_neighbors:
                if n.lattice_normal_form == original_cnf.lattice_normal_form:
                    second_neighb_xtal = n.reconstruct()
                    original_xtal = original_cnf.reconstruct()
                    print(np.array(n.coords) - np.array(original_cnf.coords))
                    print()
                    # helpers.assert_identical_by_pdd_distance(struct, original_xtal, cutoff=0.1)
                    # helpers.assert_identical_by_pdd_distance(second_neighb_xtal, original_xtal)
            break
        # assert original_cnf in second_neighbors