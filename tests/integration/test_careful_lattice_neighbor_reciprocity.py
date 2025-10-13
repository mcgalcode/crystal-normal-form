import pytest
import helpers

from cnf.crystal_normal_form import CrystalNormalForm, LatticeNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.navigation.lattice_neighbors import LatticeNeighborFinder, PermutationMatrix, LatticeStep
from cnf.lattice.permutations import ConormPermutation, MatrixTuple

def test_pathological_reciprocity_case(mp_structures):
    struct = mp_structures[293]
    # struct = struct.to_primitive()
    xi = 1.5
    delta = 30
    print(f"Struct has {len(struct)} sites!")
    
    cnf_constructor = CNFConstructor(xi, delta)
    original_cnf = cnf_constructor.from_pymatgen_structure(struct).cnf
    original_lnf = original_cnf.lattice_normal_form
    print(f"Original CNF: {original_cnf.coords}")

    neighbor_finder = LatticeNeighborFinder(original_cnf, verbose_logging=False)
    neighbors = neighbor_finder.find_cnf_neighbors().neighbors
    print(f"Structure has {len(neighbors)} neighbors")

    specific_perm = ConormPermutation((5, 4, 2, 3, 1, 0, 6))
    specific_mat = MatrixTuple.from_tuple((-1, 0, 0, -1, 0, 1, -1, 1, 0))
    specific_step = LatticeStep(
        [1, 0, -1, 0, 0, 0, 0],
        PermutationMatrix(specific_perm, specific_mat, [MatrixTuple.from_tuple((-1, 0, 0, -1, 0, 1, -1, 1, 0))])
    )
    print(F"Investigating step: {specific_step}")
    
    neighbor_finder.verbose_logging = True
    cnf_neighbor = neighbor_finder.find_cnf_neighbor(specific_step).result

    second_neighbor_finder = LatticeNeighborFinder(cnf_neighbor, verbose_logging=False)
    second_neighbors = second_neighbor_finder.find_cnf_neighbors()
    print(f"Pathological neighbor has {len(second_neighbors)} neighbors")
    for sn in second_neighbors.neighbors:
        pt = sn.point
        lnf = pt.lattice_normal_form
        if original_lnf == lnf:
            step_results = second_neighbors.steps_for_neighbor(sn.point)
            print(f"Found {len(step_results)} steps that led to a neighbor with the original LNF...")
            step = step_results[0].step
            print(step)
            print(f"Neighbor LNF was: {pt.lattice_normal_form.coords}")
            print(f"Neighbor BNF was: {pt.basis_normal_form.coord_list}")
            helpers.assert_identical_by_pdd_distance(original_cnf.reconstruct(), pt.reconstruct())
            print(f"Found that this neighbor was geometrically identical!")
    second_neighb_lnfs = [neighbor.point.lattice_normal_form for neighbor in second_neighbors.neighbors]
    assert original_lnf in second_neighb_lnfs
    assert original_cnf in second_neighbors
    return
    
    
    original_xtal = original_cnf.reconstruct()
    # helpers.assert_identical_by_pdd_distance(struct, original_xtal, cutoff=0.1)

    recipricol_cnf_nbs = []
    nonreciprocal_cnf_nbs = []

    recipricol_lnf_nbs = []
    nonreciprocal_lnf_nbs = []

    for neighbor_step in LatticeNeighborFinder(original_cnf).find_cnf_neighbors().neighbors:
        n = neighbor_step.point
        second_neighbor_set = LatticeNeighborFinder(n).find_cnf_neighbors()
        second_nb_lnfs = [nb.point.lattice_normal_form for nb in second_neighbor_set.neighbors]

        if original_lnf in second_nb_lnfs:
            recipricol_lnf_nbs.append(n)
        else:
            nonreciprocal_lnf_nbs.append(n)
        # assert original_cnf in second_neighbors
        if original_cnf not in second_neighbor_set:
            print()
            print(f"No reciprocal relationship found! For neighbor:")
            print(neighbor_step.step_results[0].step.vals)
            print(neighbor_step.step_results[0].step.prereq_perm.conorm_permutation.perm)
            print(neighbor_step.step_results[0].step.prereq_perm.matrix.tuple)
            print(neighbor_step.step_results[0].step.prereq_perm.all_matrices)

            nonreciprocal_cnf_nbs.append(n)
            # for sn in second_neighbor_set.neighbors:
            #     sn = sn.neighbor
            #     if sn.lattice_normal_form == original_cnf.lattice_normal_form:
            #         second_neighb_xtal = sn.reconstruct()
            #         # print(sn)
            #         # print(np.array(n.coords) - np.array(original_cnf.coords))
            #         print("Checking if xtals with matching LNFs are actually the same crystal")
            #         helpers.assert_identical_by_pdd_distance(second_neighb_xtal, original_xtal, cutoff=0.005)
            #         print("Manually found matching neighbor using PDD: ")
            #         print(sn)
        else:
            recipricol_cnf_nbs.append(n)

    print(f"Found {len(recipricol_cnf_nbs)} GOOD CNF neighbors")
    print(f"Found {len(nonreciprocal_cnf_nbs)} BAD CNF neighbors")
    

    print(f"Found {len(recipricol_lnf_nbs)} GOOD LNF neighbors")
    print(f"Found {len(nonreciprocal_lnf_nbs)} BAD LNF neighbors")    

# def test_pathological_specific_nb(mp_structures):
