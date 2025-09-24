import numpy as np
from ..permutations import CONORM_PERMUTATION_TO_VONORM_PERMUTATION, ConormPermutation
from .conorm_list_form import ConormListForm

class ConormList():

    def __init__(self, conorms, tol=1e-8):
        self.conorms = conorms
        self.form = ConormListForm([idx for idx, conorm in enumerate(self.conorms) if np.abs(conorm) < tol])
        self.permissible_permutations = self._compute_permissible_permutations()

    def apply_permutation(self, permutation: tuple):
        permuted_vals = []
        for p in permutation:
            if p < 6:
                permuted_vals.append(self.conorms[p])
            else:
                permuted_vals.append(0)
        return ConormList(tuple(permuted_vals[:6]))

    def is_permutation_permissible(self, permutation):
        # return permutation[-1] == 6
        return permutation[-1] in self.form.zero_indices or permutation[-1] == 6
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