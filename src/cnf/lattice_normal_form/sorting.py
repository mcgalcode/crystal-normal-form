import copy


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

def swap_vonorm_idxs_in_place(idx1, idx2, vonorms):
    if not (idx1 < idx2):
        raise RuntimeError(f"Out-of-order vonorm idx pair ({idx1}, {idx2})  swap requested (idx1 must be less than idx2)")

    if is_valid_swap(idx1, idx2, vonorms):
        # print(f"Swapping: {idx1}, {idx2}")
        secondary_swap = primary_swap_to_secondary_swap[(idx1, idx2)]
        swap_list_items_in_place(idx1, idx2, vonorms)
        swap_list_items_in_place(secondary_swap[0], secondary_swap[1], vonorms)
    else:
        raise RuntimeError(f"Could not swap vonorms {idx1} and{idx2}! Indices must be primary vonorms.")    
    
def swap_list_items_in_place(idx1, idx2, items):
    tmp = items[idx2]
    items[idx2] = items[idx1]
    items[idx1] = tmp

def sort_vonorms(vonorms):
    primary_vonorms_sorted, out_of_order_pair = check_primary_vonorms_sorted(vonorms)
    
    while not primary_vonorms_sorted:
        swap_vonorm_idxs_in_place(out_of_order_pair[0], out_of_order_pair[1], vonorms)
        primary_vonorms_sorted, out_of_order_pair = check_primary_vonorms_sorted(vonorms)
    
    for idx1 in range(4,6):
        for idx2 in range(idx1, 7):
            if vonorms[idx1] > vonorms[idx2]:
                # print(f"Found out of order secondary vonorms! {idx1}, {idx2}")
                swaps = get_possible_swaps((idx1, idx2), vonorms)
                if len(swaps) > 0:
                    swap_vonorm_idxs_in_place(swaps[0][0], swaps[0][1], vonorms)

