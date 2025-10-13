import numpy as np
from ..swaps.sorting import swap_vonorm_idxs
from .conorm_list import ConormList
from ...linalg import MatrixTuple
from ..permutations import apply_permutation, Permutation, ConormPermutation, VonormPermutation, PermutationMatrix

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
        if not (isinstance(vonorms, tuple) or isinstance(vonorms, list) or isinstance(vonorms, np.ndarray)):
            raise ValueError(f"Tried to intialize VonormList with bad type {type(vonorms)}")
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
    
    def has_same_members(self, other: 'VonormList', tol=1e-8):
        diff = np.abs(np.array(sorted(self.vonorms)) - np.array(sorted(other.vonorms)))
        return np.all(diff < tol)
    
    def about_equal(self, other: 'VonormList', tol=1e-8):
        diff = np.abs(np.array(self.vonorms) - np.array(other.vonorms))
        return np.all(diff < tol)
    
    def canonical_matrix_for_perm(self, perm: Permutation):
        return self.conorms.canonical_matrix_for_perm(perm)

    def apply_permutation(self, permutation: tuple):
        return VonormList(tuple(apply_permutation(self.vonorms, permutation)))
    
    def stabilizer_perms(self, tol=1e-8) -> list[PermutationMatrix]:
        possible_perms = self.conorms.permissible_permutations
        stabilizers = []
        for p in possible_perms:
            vonorm_perm = p.vonorm_permutation
            permuted_self = self.apply_permutation(vonorm_perm)
            if self.about_equal(permuted_self, tol):
                stabilizers.append(p)
        return stabilizers

    def stabilizer_matrices(self, tol=1e-8) -> list[MatrixTuple]:
        perm_stab = self.stabilizer_perms(tol)
        mats = []
        for p in perm_stab:
            mats.extend(p.all_matrices)
        return list(set(mats))

    def to_generators(self, lattice_step_size: float):
        physical_vonorms = np.array(self.vonorms) * lattice_step_size
        physical_conorms = np.array(self.conorms.conorms) * lattice_step_size

        # A near-zero vonorm (squared norm) indicates a degenerate lattice.
        if np.any(np.isclose(physical_vonorms[:3], 0.0)):
            raise ValueError("Lattice vectors cannot have zero length (vonorms are close to zero).")

        # MODIFICATION: Add a check for physically meaningless negative squared norms.
        if np.any(physical_vonorms[:3] < 0.0):
            raise ValueError("Lattice vectors cannot have negative squared norms (vonorms are negative).")

        v0_dot_v1 = physical_conorms[0]
        v0_dot_v2 = physical_conorms[1]
        v1_dot_v2 = physical_conorms[3]

        v0_norm = np.sqrt(physical_vonorms[0])
        v1_norm = np.sqrt(physical_vonorms[1])
        v2_norm = np.sqrt(physical_vonorms[2])

        cos_x = v0_dot_v1 / (v0_norm * v1_norm)    
        # MODIFICATION: Clip the value to the valid cosine range [-1, 1].
        x = np.clip(cos_x, -1.0, 1.0)
        y = np.sqrt(np.maximum(0.0, 1 - x ** 2))

        cos_a = v0_dot_v2 / (v0_norm * v2_norm)
        a = np.clip(cos_a, -1.0, 1.0)
        
        if y > 1e-9:
            # The denominator check is no longer needed due to the validation above.
            term_for_b = v1_dot_v2 / (v1_norm * v2_norm)
            # The original formula had a slight error in grouping, this is the standard Gram-Schmidt term.
            b = (1 / y) * (term_for_b - x * a)
        else:
            b = 0.0

        c_squared = 1 - a ** 2 - b ** 2
        c = np.sqrt(np.maximum(0.0, c_squared))

        v0 = v0_norm * np.array([1, 0, 0])
        v1 = v1_norm * np.array([x, y, 0])
        v2 = v2_norm * np.array([a, b, c])
        return np.array([v0, v1, v2])
    
    def to_superbasis(self, lattice_step_size: float = 1.0):
        from ..superbasis import Superbasis
        return Superbasis.from_generating_vecs(self.to_generators(lattice_step_size=lattice_step_size))

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