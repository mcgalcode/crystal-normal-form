import enum
import numpy as np
import copy
from .sorting import swap_vonorm_idxs
from .conorm_list import ConormList
from .permutations import apply_permutation

# This matrix is found on page 48 of David's thesis
VONORM_TO_DOT_PRODUCTS = np.array([
    [-1, -1, 0, 0, 1, 0],
    [-1, 0, -1, 0, 0, 1],
    [0, 1, 1, 0, -1, -1],
    [1, 0, 0, 1, -1, -1],
    [0, -1, 0, -1, 0, 1],
    [0, 0, -1, -1, 1, 0]
])

class Conorms(enum.Enum):

    P_01 = 0
    P_02 = 1
    P_03 = 2
    P_12 = 3
    P_13 = 4
    P_23 = 5

CONORM_IDX_TO_VECTOR_PAIRS = {
    0: (0, 1),
    1: (0, 2),
    2: (0, 3),
    3: (1, 2),
    4: (1, 3),
    5: (2, 3),
}

VECTOR_PAIRS_TO_CONORM_IDXS = { v: k for k, v in CONORM_IDX_TO_VECTOR_PAIRS.items() }

ALL_INDICES = {0, 1, 2, 3}

SECONDARY_VONORM_LABELS_TO_IDXS = {
    (0, 1): 4,
    (2, 3): 4,
    (0, 2): 5,
    (1, 3): 5,
    (0, 3): 6,
    (1, 2): 6,
}

def to_canonical_pair(pair):
    pair = list(pair)
    return tuple(sorted(pair))

class VonormList():

    def __init__(self, vonorms):
        self.vonorms = vonorms

    @property
    def conorms(self):
        return ConormList((1 / 2) * VONORM_TO_DOT_PRODUCTS @ self.vonorms[:6])
    
    def is_obtuse(self, tol=0):
        return all([c <= tol for c in self.conorms])
    
    def __getitem__(self, key):
        return self.vonorms[key]
    
    def swap_labels(self, swap_pair, return_swaps=False):
        i, j = swap_pair
        if return_swaps:
            new_vonorms, swaps = swap_vonorm_idxs(i, j, self.vonorms, in_place=False, return_swaps=True)
            return VonormList(new_vonorms), swaps
        else:
            new_vonorms = swap_vonorm_idxs(i, j, self.vonorms, in_place=False)
            return VonormList(new_vonorms)
    
    def has_same_members(self, other: 'VonormList', decimal_comparison=3):
        this = tuple(sorted(list(np.round(self.vonorms, decimal_comparison))))
        that = tuple(sorted(list(np.round(other.vonorms, decimal_comparison))))
        return this == that

    def apply_permutation(self, permutation: tuple):
        return VonormList(tuple(apply_permutation(self.vonorms, permutation)))
    
    def selling_transform(self) -> tuple["VonormList", tuple[int, int]]:
        positive_conorm_indices = [i for i, conorm in enumerate(self.conorms) if conorm > 0]
        first_idx = positive_conorm_indices[0]
        acute_vector_pair = CONORM_IDX_TO_VECTOR_PAIRS[first_idx]
        i, j = tuple(acute_vector_pair)
        k, l = tuple(ALL_INDICES - set(acute_vector_pair))

        new_vonorm_list = [0, 0, 0, 0, 0, 0, 0]

        # Following Kurlin Lemma A.1
        # Two vonorms remain the same:
        new_vonorm_list[i] = self.vonorms[i]
        new_vonorm_list[j] = self.vonorms[j]

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
        ik_idx = SECONDARY_VONORM_LABELS_TO_IDXS[to_canonical_pair({i, k})]
        new_vonorm_list[k] = self.vonorms[ik_idx]
        new_vonorm_list[ik_idx] = self.vonorms[k]

        # pair 2: u_l = v_il, u_il = u_jk = v_l
        il_idx = SECONDARY_VONORM_LABELS_TO_IDXS[to_canonical_pair({i, l})]
        new_vonorm_list[l] = self.vonorms[il_idx]
        new_vonorm_list[il_idx] = self.vonorms[l]

        # The i,j vonorm is reduced by 4 x v_i dot v_j
        vector_pair = to_canonical_pair({i, j})
        ij_idx = SECONDARY_VONORM_LABELS_TO_IDXS[vector_pair]
        conorm_v_i_dot_v_j = self.conorms[VECTOR_PAIRS_TO_CONORM_IDXS[vector_pair]]
        new_vonorm_list[ij_idx] = self.vonorms[ij_idx] - 4 * conorm_v_i_dot_v_j
        return VonormList(new_vonorm_list), acute_vector_pair

    def to_generators(self, epsilon: float):
        v0_dot_v1, v0_dot_v2, _, v1_dot_v2, _, _ = self.conorms # * epsilon

        v0_norm = np.sqrt(self[0] * epsilon)
        v1_norm = np.sqrt(self[1] * epsilon)
        v2_norm = np.sqrt(self[2] * epsilon)

        x = v0_dot_v1 / (epsilon ** 2 * v0_norm * v1_norm) # Max
        # x = v0_dot_v1 * epsilon / (v0_norm * v1_norm) # David

        y = np.sqrt(1 - x ** 2) # Max + David

        a = v0_dot_v2 / (epsilon ** 2 * v0_norm * v2_norm) # Max
        # a = v0_dot_v2 * epsilon / (v0_norm * v2_norm) # David
        
        b = (1 / y) * (v1_dot_v2 / (epsilon ** 2 * v1_norm * v2_norm) - x * a) # Max
        # b = (1 / y) * (v1_dot_v2 * epsilon / (v1_norm * v2_norm) - x * a) # David

        c = np.sqrt(1 - a ** 2 - b ** 2)

        v0 = epsilon * v0_norm * np.array([1, 0, 0])
        v1 = epsilon * v1_norm * np.array([x, y, 0])
        v2 = epsilon * v2_norm * np.array([a, b, c])
        return np.array([v0, v1, v2])
    
    def to_superbasis(self, epsilon: float = 1.0):
        from .superbasis import Superbasis
        return Superbasis.from_generating_vecs(self.to_generators(epsilon=epsilon))


    def __repr__(self):
        numbers = " ".join([str(v) for v in self.vonorms])
        return f"Vonorms({numbers})"
        
    def __eq__(self, other: "VonormList"):
        return np.all(np.isclose(self.vonorms, other.vonorms))
    
    def __hash__(self):
        return tuple(self.vonorms).__hash__()
    
    def __getitem__(self, key):
        return self.vonorms[key]