import numpy as np
from ..permutations import Permutation, VonormPermutation
from .coform import Coform
from ...linalg import MatrixTuple

class ConormList():

    def __init__(self, conorms, tol=1e-3):
        self.conorms = conorms
        self.form = Coform([idx for idx, conorm in enumerate(self.conorms) if np.abs(conorm) < tol])

    def set_tol(self, new_tol):
        return ConormList(self.conorms, new_tol)

    @property
    def permissible_permutations(self):
        return self.form.permissible_permutations()

    def apply_permutation(self, permutation: tuple):
        permuted_vals = []
        for p in permutation:
            if p < 6:
                permuted_vals.append(self.conorms[p])
            else:
                permuted_vals.append(0)
        return ConormList(tuple(permuted_vals[:6]))
    
    def all_permutation_matrices(self) -> list[MatrixTuple]:
        mats = []
        for perm in self.permissible_permutations:
            mats.extend(perm.all_matrices)
        return list(set(mats))

    def is_permutation_permissible(self, permutation):
        # return permutation[-1] == 6
        return permutation[-1] in self.form.zero_indices or permutation[-1] == 6
        # vperm = ConormPermutation(permutation).to_vonorm_permutation()
        # return 4 not in vperm[:4] and 5 not in vperm[:4] and 6 not in vperm[:4]

    def has_same_members(self, other: 'ConormList', tol=1e-8):
        diff = np.abs(np.array(sorted(self.conorms)) - np.array(sorted(other.conorms)))
        return np.all(diff < tol)
    
    def canonical_matrix_for_perm(self, perm: Permutation):
        if isinstance(perm, VonormPermutation):
            return self.form.canonical_matrix_for_perm(perm.to_conorm_permutation())
        else:
            return self.form.canonical_matrix_for_perm(perm)
    
    def about_equal(self, other: 'ConormList', tol=1e-8):
        diff = np.abs(np.array(self.conorms) - np.array(other.conorms))
        return np.all(diff < tol)

    def __iter__(self):
        return iter(self.conorms)
    
    def __eq__(self, other: "ConormList"):
        return np.all(np.isclose(self.conorms, other.conorms))
 
    def __getitem__(self, key):
        return self.conorms[key]
    
    def __repr__(self):
        numbers = " ".join([str(v) for v in self.conorms])
        return f"Conorms({numbers})"