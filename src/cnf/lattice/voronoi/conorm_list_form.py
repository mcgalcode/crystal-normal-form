from .voronoi_values import Conorm
from .constants import CONORM_IDX_TO_PAIR
from ..permutations import UnimodPermMapper, ConormPermutation, PermutationMatrix
from ...linalg import MatrixTuple
from ...utils.sorted_tuple import SortedTuple
from itertools import combinations

import random
import numpy as np

def perm_in_s4(perm):
    return perm[4:] == (4, 5, 6)

# Note that for Voronoi class V5 requires that 3 conorms are zero. This
# corresponds to a cuboid voronoi cell (i.e. 3 pairs of lattice vectors
# are orthogonal to each other)
#
# There are several sets of 3 conorms that cannot be simultaneously zero.
# These correspond to situations where a single voronoi vector is orthogonal
# to all three zeros. For example,
#
# (0, 1, 2) -> p01, p02, p03 : V0 is orthogonal to all of v1, v2, and v3
#
# This is only possible of v1, v2, and v3 are coplanar. If v1, v2, and v3 are
# coplanar, then the set {v0, v1, v3, v3} is not a superbasis (it does not
# satisfy v0 = -v1 -v2 -v3)

IMPOSSIBLE_ZERO_SETS = [
    (0, 1, 2),
    (0, 3, 4),
    (1, 3, 5),
    (2, 4, 5),
]

class ConormListForm():

    @staticmethod
    def all_coforms() -> list['ConormListForm']:
        all_conorm_idxs = range(0,6)
        result = []
        for num_zeros in range(0, 4):
            idx_combos = combinations(all_conorm_idxs, num_zeros)
            for c in idx_combos:
                if c not in IMPOSSIBLE_ZERO_SETS:
                    result.append(ConormListForm(c))
        return result

    @staticmethod
    def get_coforms_of_voronoi_class(voronoi_class: int) -> list['ConormListForm']:
        if not voronoi_class in range(1,6):
            raise ValueError(f"Requested coforms for invalid voronoi class: {voronoi_class} (not in 1-5)")
        return [cf for cf in ConormListForm.all_coforms() if cf.voronoi_class == voronoi_class]

    @staticmethod
    def all_grouped_vonorm_permutations() -> list[list[PermutationMatrix]]:
        groups = {}
        for cf in ConormListForm.all_coforms():
            for p in cf.permissible_permutations():
                s4_members = tuple(sorted(p.vonorm_permutation.perm[:4]))
                if s4_members in groups:
                    groups[s4_members].append(p)
                else:
                    groups[s4_members] = [p]
                groups[s4_members] = sorted(groups[s4_members], key=lambda p: p.vonorm_permutation)
        return groups

    def __init__(self, zero_indices):
        self.zero_indices = SortedTuple(*zero_indices)
        self.voronoi_class = self._compute_voronoi_class()
        self.tuple = tuple(zero_indices)
    
    def permissible_permutations(self) -> list[PermutationMatrix]:
        perms = UnimodPermMapper.get_perms_for_zero_set(self.zero_indices)
        # perms = [p for p in perms if perm_in_s4(p)]
        return [self.build_perm_matrix(p) for p in perms]
    
    def build_perm_matrix(self, perm):
        return PermutationMatrix(
            perm,
            self.canonical_matrix_for_perm(perm),
            self.matrices_for_perm(perm)
        )
    
    def similar_coforms(self):
        return ConormListForm.get_coforms_of_voronoi_class(self.voronoi_class)
    
    def all_matrices(self):
        mats = []
        for p in self.permissible_permutations():
            mats.extend(p.all_matrices)
        return list(set(mats))

    def all_matrices_for_similar_coforms(self):
        mats = []
        for cf in self.similar_coforms():
            for p in cf.permissible_permutations():
                mats.extend(p.all_matrices)
        return list(set(mats))
    
    def grouped_vonorm_permutations(self) -> list[list[PermutationMatrix]]:
        groups = {}
        for p in self.permissible_permutations():
            s4_members = tuple(sorted(p.vonorm_permutation.perm[:4]))
            if s4_members in groups:
                groups[s4_members].append(p)
            else:
                groups[s4_members] = [p]
            groups[s4_members] = sorted(groups[s4_members], key=lambda p: p.vonorm_permutation)            
        return groups
    
    def matrices_for_perm(self, cperm: ConormPermutation):
        mats = UnimodPermMapper.get_matrices_for_zero_set_and_perm(self.zero_indices, cperm)
        return mats
    
    def canonical_matrix_for_perm(self, cperm: ConormPermutation):
        # NOTE: ONLY CONSIDER A SINGLE ONE OF THESE MATRICES
        # Choose the one that is closest to the identity
        i = np.eye(3)
        def _get_diff(mat: MatrixTuple):
            return np.linalg.norm(i - mat.matrix)
        mats = sorted(self.matrices_for_perm(cperm), key=_get_diff)
        return mats[0]

    def zero_conorms(self):
        return [Conorm.from_idx(idx) for idx in self.zero_indices]

    def _compute_voronoi_class(self):
        if len(self.zero_indices) == 3:
            return 5
        
        if len(self.zero_indices) == 1:
            return 2
        
        if len(self.zero_indices) == 0:
            return 1
        
        if len(self.zero_indices) == 2:
            pair_1 = self.zero_indices[0]
            pair_2 = self.zero_indices[1]
            if len(set(CONORM_IDX_TO_PAIR[pair_1]).intersection(set(CONORM_IDX_TO_PAIR[pair_2]))) == 0:
                return 3
            else:
                return 4

    def __repr__(self):
        return f"ConormListForm({self.zero_indices})"
    
    def __len__(self):
        return len(self.zero_indices)