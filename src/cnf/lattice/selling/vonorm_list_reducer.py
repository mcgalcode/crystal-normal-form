from .selling_reducer import SellingReducer
from ..vonorm_list import VonormList
from .selling_pair import SellingPair

class VonormListSellingReducer(SellingReducer):
 
    CONORM_IDX_TO_VECTOR_PAIRS = {
        0: (0, 1),
        1: (0, 2),
        2: (0, 3),
        3: (1, 2),
        4: (1, 3),
        5: (2, 3),
    }

    SECONDARY_VONORM_LABELS_TO_IDXS = {
        (0, 1): 4,
        (2, 3): 4,
        (0, 2): 5,
        (1, 3): 5,
        (0, 3): 6,
        (1, 2): 6,
    }

    ALL_INDICES = {0, 1, 2, 3}

    VECTOR_PAIRS_TO_CONORM_IDXS = { v: k for k, v in CONORM_IDX_TO_VECTOR_PAIRS.items() }

    def _logging_repr(self, obj: VonormList):
        return f"{obj} {obj.conorms}"

    
    def apply_selling_transform(self, obj: VonormList):
        positive_conorm_indices = [i for i, conorm in enumerate(obj.conorms) if conorm > 0]
        selected_conorm_idx = positive_conorm_indices[0]
        acute_vector_pair = VonormListSellingReducer.CONORM_IDX_TO_VECTOR_PAIRS[selected_conorm_idx]
        i, j = tuple(acute_vector_pair)
        k, l = tuple(VonormListSellingReducer.ALL_INDICES - set(acute_vector_pair))

        new_vonorm_list = [0, 0, 0, 0, 0, 0, 0]

        # Following Kurlin Lemma A.1
        # Two vonorms remain the same:
        new_vonorm_list[i] = obj.vonorms[i]
        new_vonorm_list[j] = obj.vonorms[j]

        # Two vonorm pairs swap:
        #
        # This is trickier: v_ik is a secondary vonorm, but there are two ways
        # that each secondary vonorm is expressed: v0 + v1 = -v2 - v3
        #                                       (v0 +v1)^2 = (v2 + v3)^2
        # So if i and k are 2 and 3, then v_ik isn't v_23, which isn't a label
        # we use - it's v_01. The map SECONDARY_VONORM_LABELS_TO_IDXS enumerates these
        # relationships and makes it easy to grab the index corresponding to a given ik pair
        #
        # pair 1: u_k = v_ik, u_ik = u_jl = v_k
        ik_idx = VonormListSellingReducer.SECONDARY_VONORM_LABELS_TO_IDXS[SellingPair(i, k)]
        new_vonorm_list[k] = obj.vonorms[ik_idx]
        new_vonorm_list[ik_idx] = obj.vonorms[k]

        # pair 2: u_l = v_il, u_il = u_jk = v_l
        il_idx = VonormListSellingReducer.SECONDARY_VONORM_LABELS_TO_IDXS[SellingPair(i, l)]
        new_vonorm_list[l] = obj.vonorms[il_idx]
        new_vonorm_list[il_idx] = obj.vonorms[l]

        # The i,j vonorm is reduced by 4 x v_i dot v_j
        vector_pair = SellingPair(i, j)
        ij_idx = VonormListSellingReducer.SECONDARY_VONORM_LABELS_TO_IDXS[vector_pair]
        # assert VonormListSellingReducer.VECTOR_PAIRS_TO_CONORM_IDXS[vector_pair] == selected_conorm_idx
        conorm_v_i_dot_v_j = obj.conorms[selected_conorm_idx]
        new_vonorm_list[ij_idx] = obj.vonorms[ij_idx] - 4 * conorm_v_i_dot_v_j
        print(f"Reducing vonorm {ij_idx} ({obj.vonorms[ij_idx]} -> {new_vonorm_list[ij_idx]}) using conorm idx {selected_conorm_idx}, w value {conorm_v_i_dot_v_j}")
        if isinstance(obj.vonorms[ij_idx], int):
            new_vonorm_list[ij_idx] = int(new_vonorm_list[ij_idx])

        return VonormList(new_vonorm_list), acute_vector_pair
