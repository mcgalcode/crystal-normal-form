import copy
import numpy as np

from .vonorm_list import VonormList
from .sorting import apply_swap_series 
from ..lattice_normal_form.unimodular import get_unimodular_matrix_from_voronoi_vector_idxs


def search_for_stabilizers(vlist: VonormList,
                           starting_permutation = None, 
                           explored_permutations = None,
                           stabilizer_permutations = None,
                           transform_matrix = None):
    # Implements a depth first search
    
    if starting_permutation is None:
        starting_permutation = list(range(0,7))
        stabilizer_permutations = {
            tuple(starting_permutation): np.eye(3, dtype=np.int64)
        }
    
    if explored_permutations is None:
        explored_permutations = set([tuple(range(0,7))])
    
    if transform_matrix is None:
        transform_matrix = np.eye(3)

    possible_swaps = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3)
    ]
    
    for swap in possible_swaps:
        swapped_permutation = copy.copy(starting_permutation)
        swapped_vlist, swaps = vlist.swap_labels(swap, return_swaps=True)
        swapped_permutation = apply_swap_series(swapped_permutation, swaps)
        permutation_tuple = tuple(swapped_permutation)

        # If we have already looked at this permutation, skip it
        if permutation_tuple in explored_permutations:
            continue

        # print(f"Exploring {swapped_permutation} from {starting_permutation}")
        # Otherwise, mark it as explored
        explored_permutations.add(permutation_tuple)

        # If this swap yields a different vonorm list, skip it
        if swapped_vlist != vlist:
            continue
        # print(f"Found stabilizer: {permutation_tuple}")

        # If this swap does NOT change the vonorm list, add the new
        # permutation
        permutation_transform_mat = get_unimodular_matrix_from_voronoi_vector_idxs(permutation_tuple[1:4])
        # Is this a right or left matrix multiply?
        total_transform_mat = transform_matrix @ permutation_transform_mat
        stabilizer_permutations[permutation_tuple] = total_transform_mat

        # Then continue the exploration from here:
        search_for_stabilizers(
            vlist,
            swapped_permutation,
            explored_permutations,
            stabilizer_permutations,
            transform_matrix=total_transform_mat
        )
    
    return stabilizer_permutations, explored_permutations




