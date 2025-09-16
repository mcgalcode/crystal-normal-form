import copy
from .unimodular import get_unimodular_matrix_for_swap_series

def check_primary_vonorms_sorted(vonorms):
    for idx in range(0,3):
        if vonorms[idx] > vonorms[idx + 1]:
            return False, (idx, idx + 1)
    return True, None

def check_secondary_vonorms_sorted(vonorms):
    for idx1 in range(4,6):
        for idx2 in range(idx1, 6):
            if vonorms[idx1] > vonorms[idx2]:
                return False, (idx1, idx2)
    return True, None


def is_primary_vonorm_idx(idx):
    return idx < 4

def is_secondary_vonorm_idx(idx):
    return not is_primary_vonorm_idx(idx)

primary_swap_to_secondary_swap = {
    (0,1): (5,6),
    (0,2): (4,6),
    (0,3): (4,5),
    (1,2): (4,5),
    (1,3): (4,6),
    (2,3): (5,6)
}

secondary_swap_to_primary_swaps = {}

for primary_swap, secondary_swap in primary_swap_to_secondary_swap.items():
    if secondary_swap in secondary_swap_to_primary_swaps:
        secondary_swap_to_primary_swaps[secondary_swap].append(primary_swap)
    else:
        secondary_swap_to_primary_swaps[secondary_swap] = [primary_swap]

def get_possible_swaps(secondary_idx_pair, vonorm_list):
    relevant_primary_swaps = secondary_swap_to_primary_swaps[
        secondary_idx_pair
    ]
    swappable_primary_pairs = []
    for pair in relevant_primary_swaps:
        primary_idx_1, primary_idx_2 = pair
        if vonorm_list[primary_idx_1] == vonorm_list[primary_idx_2]:
            swappable_primary_pairs.append(pair)

    return swappable_primary_pairs

def is_valid_swap(idx1, idx2, vonorms):
    # In the future, this will depend on the symmetry
    # of the lattice.
    return is_primary_vonorm_idx(idx1) and is_primary_vonorm_idx(idx2)

def swap_vonorm_idxs(idx1, idx2, vonorms, in_place=True, return_swaps=False):
    if in_place and isinstance(vonorms, tuple):
        raise ValueError("Cannot swap vonorms in place when represented as tuple!")
    
    if not (idx1 < idx2):
        raise RuntimeError(f"Out-of-order vonorm idx pair ({idx1}, {idx2})  swap requested (idx1 must be less than idx2)")

    if is_valid_swap(idx1, idx2, vonorms):
        # print(f"Swapping: {idx1}, {idx2}")
        secondary_swap = primary_swap_to_secondary_swap[(idx1, idx2)]
        if in_place == True:
            swap_list_items_in_place(idx1, idx2, vonorms)
            swap_list_items_in_place(secondary_swap[0], secondary_swap[1], vonorms)
        else:
            vonorms = swap_list_items(idx1, idx2, vonorms)
            vonorms = swap_list_items(secondary_swap[0], secondary_swap[1], vonorms)
        
        if return_swaps:
            return vonorms, [(idx1, idx2), secondary_swap]
        else:
            return vonorms
    else:
        raise RuntimeError(f"Could not swap vonorms {idx1} and{idx2}! Indices must be primary vonorms.")  
    
def swap_list_items_in_place(idx1, idx2, items):
    tmp = items[idx2]
    items[idx2] = items[idx1]
    items[idx1] = tmp

def swap_list_items(idx1, idx2, items):
    copied_items = list(copy.copy(items))
    copied_items[idx1] = items[idx2]
    copied_items[idx2] = items[idx1]
    return tuple(copied_items)

def apply_swap_series(items, swaps):
    items = copy.copy(items)
    for swap in swaps:
        i, j = swap
        swap_list_items_in_place(i, j, items)
    return items
        
def sort_vonorms(vonorms, return_transform_mat=False) -> list[tuple[int, int]]:
    swaps = []
    primary_vonorms_sorted, out_of_order_pair = check_primary_vonorms_sorted(vonorms)
    
    while not primary_vonorms_sorted:
        print(out_of_order_pair)
        vonorms = swap_vonorm_idxs(out_of_order_pair[0], out_of_order_pair[1], vonorms)
        print("After: ", out_of_order_pair)
        swaps.append(out_of_order_pair)
        print("Swaps so far: ", swaps)
        primary_vonorms_sorted, out_of_order_pair = check_primary_vonorms_sorted(vonorms)
    
    for idx1 in range(4,6):
        for idx2 in range(idx1, 7):
            if vonorms[idx1] > vonorms[idx2]:
                # print(f"Found out of order secondary vonorms! {idx1}, {idx2}")
                possible_swaps = get_possible_swaps((idx1, idx2), vonorms)
                # There may be multiple possible swaps that can achieve this
                # reordering, but we just need to execute one of them.
                if len(swaps) > 0:
                    selected_swap = possible_swaps[0]
                    vonorms = swap_vonorm_idxs(selected_swap[0], selected_swap[1], vonorms)
                    swaps.append(selected_swap)
    if return_transform_mat:
        return swaps, get_unimodular_matrix_for_swap_series(swaps)
    else:
        return swaps

