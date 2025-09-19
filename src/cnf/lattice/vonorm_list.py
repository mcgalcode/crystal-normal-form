import enum
import numpy as np
import copy
from .swaps.sorting import swap_vonorm_idxs
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

    def primary_sum(self):
        return sum(self.vonorms[:4])

    def secondary_sum(self):
        return sum(self.vonorms[4:])
    
    def is_superbasis(self):
        return np.isclose(self.primary_sum(), self.secondary_sum())
    
    @property
    def tuple(self):
        return tuple(self.vonorms)
    
    def __repr__(self):
        numbers = " ".join([str(v) for v in self.vonorms])
        return f"Vonorms({numbers})"
        
    def __eq__(self, other: "VonormList"):
        return np.all(np.isclose(self.vonorms, other.vonorms))
    
    def __hash__(self):
        return self.tuple.__hash__()
    
    def __getitem__(self, key):
        return self.vonorms[key]