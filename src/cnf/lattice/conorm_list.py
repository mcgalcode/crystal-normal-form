import enum
from .permutations import CONORM_PERMUTATION_TO_VONORM_PERMUTATION, ConormPermutation, VonormPermutation

class Conorms(enum.Enum):

    P_01 = 0
    P_02 = 1
    P_03 = 2
    P_12 = 3
    P_13 = 4
    P_23 = 5


conorm_swaps = {
    (0, 1): [(Conorms.P_02, Conorms.P_12), (Conorms.P_03, Conorms.P_13)],
    (0, 2): [(Conorms.P_01, Conorms.P_12), (Conorms.P_03, Conorms.P_23)],
    (0, 3): [(Conorms.P_02, Conorms.P_23), (Conorms.P_01, Conorms.P_13)],
    (1, 2): [(Conorms.P_01, Conorms.P_02), (Conorms.P_13, Conorms.P_23)],
    (1, 3): [(Conorms.P_01, Conorms.P_03), (Conorms.P_12, Conorms.P_23)],
    (2, 3): [(Conorms.P_02, Conorms.P_03), (Conorms.P_12, Conorms.P_13)],
}

CONORM_INDICES_TO_PAIRS = {
    0: {0,1},
    1: {0,2},
    2: {0,3},
    3: {1,2},
    4: {1,3},
    5: {2,3}
}

class ConormList():

    def __init__(self, conorms):
        self.conorms = conorms
        self.zero_indices = [idx for idx, conorm in enumerate(self.conorms) if conorm == 0]
        self.voronoi_class = self._compute_voronoi_class()
        self.permissible_permutations = self._compute_permissible_permutations()

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
            if len(CONORM_INDICES_TO_PAIRS[pair_1].intersection(CONORM_INDICES_TO_PAIRS[pair_2])) == 0:
                return 3
            else:
                return 4
    
    def is_permutation_permissible(self, permutation):
        return permutation[-1] in self.zero_indices or permutation[-1] == 6
        # vperm = ConormPermutation(permutation).to_vonorm_permutation()
        # return 4 not in vperm[:4] and 5 not in vperm[:4] and 6 not in vperm[:4]
    
    def _compute_permissible_permutations(self) -> list[ConormPermutation]:
        permissible_perms = []
        for p in CONORM_PERMUTATION_TO_VONORM_PERMUTATION:
            if self.is_permutation_permissible(p):
                permissible_perms.append(ConormPermutation(p))
        return permissible_perms

    def __iter__(self):
        return iter(self.conorms)
 
    def __getitem__(self, key):
        return self.conorms[key]
    
    def __repr__(self):
        numbers = " ".join([str(v) for v in self.conorms])
        return f"Conorms({numbers})"