from .voronoi_values import Conorm
from .constants import CONORM_IDX_TO_PAIR
from ..permutations import UnimodPermMapper, ConormPermutation, PermutationMatrices
from ...linalg import MatrixTuple
from ...utils.sorted_tuple import SortedTuple
from itertools import combinations

class ConormListForm():

    @staticmethod
    def all_coforms() -> list['ConormListForm']:
        all_conorm_idxs = range(0,6)
        result = []
        for num_zeros in range(0, 4):
            idx_combos = combinations(all_conorm_idxs, num_zeros)
            for c in idx_combos:
                result.append(ConormListForm(c))
        return result


    def __init__(self, zero_indices):
        self.zero_indices = SortedTuple(*zero_indices)
        self.voronoi_class = self._compute_voronoi_class()
        self.tuple = tuple(zero_indices)
    
    def permissible_permutations(self) -> list[PermutationMatrices]:
        perms = UnimodPermMapper.get_perms_for_zero_set(self.zero_indices)
        return [PermutationMatrices(p, self.matrices_for_perm(p)) for p in perms]
    
    def matrices_for_perm(self, cperm: ConormPermutation):
        mats = UnimodPermMapper.get_matrices_for_zero_set_and_perm(self.zero_indices, cperm)
        return mats

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