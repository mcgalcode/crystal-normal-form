import numpy as np
from ..swaps.sorting import swap_vonorm_idxs
from .conorm_list import ConormList
from ..permutations import apply_permutation

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
    
    def has_same_members(self, other: 'VonormList', tol=1e-3):
        diff = np.abs(np.array(sorted(self.vonorms)) - np.array(sorted(other.vonorms)))
        return np.all(diff < tol)

    def apply_permutation(self, permutation: tuple):
        return VonormList(tuple(apply_permutation(self.vonorms, permutation)))

    def recover_generators(self, lattice_step_size: float = 1.0):
        """
        Recovers a set of 3 lattice generators from the 7 vonorm values.

        This method uses the vectors v1, v2, and v3 from the obtuse superbasis
        {v0, v1, v2, v3} as the lattice generators, as they form a valid
        linearly independent basis for a 3D lattice.

        Their norms and dot products are calculated from the 7 vonorms, and then
        the generator vectors are constructed in a Cartesian coordinate system.

        Args:
            lattice_step_size (float): The discretization step size used to create
                                       the integer vonorm list. Defaults to 1.0 if
                                       vonorms are already undiscretized.

        Returns:
            A 3x3 numpy array where each row is a generator vector.
        """
        # Step 1: Undiscretize the vonorms to get the squared lengths
        v0_sq, v1_sq, v2_sq, v3_sq, v01_sq, v02_sq, v03_sq = np.array(self.vonorms) * lattice_step_size

        # Step 2: Calculate the required dot products from the squared lengths.
        # These formulas are derived directly from the superbasis property (v0+v1+v2+v3=0)
        # and are more robust than using a pre-computed matrix which may contain errors.
        v1_dot_v2 = 0.5 * (v0_sq + v3_sq - v01_sq - v02_sq)
        v1_dot_v3 = 0.5 * (v0_sq + v2_sq - v01_sq - v03_sq)
        v2_dot_v3 = 0.5 * (v0_sq + v1_sq - v02_sq - v03_sq)

        # Step 3: Get the norms (lengths) of the generator vectors
        v1_norm = np.sqrt(v1_sq)

        # Ensure the first generator has a non-zero length to build the basis
        if np.isclose(v1_norm, 0):
            raise ValueError("Generator v1 cannot have zero length.")

        # Step 4: Construct the generator vectors in a Cartesian coordinate system.
        # This process is equivalent to a Cholesky decomposition of the Gram matrix.
        # It places v1 along the x-axis, v2 in the xy-plane, and v3 in 3D space.

        # Generator 1 (v1) is aligned with the x-axis
        gen_1 = np.array([v1_norm, 0.0, 0.0])

        # Generator 2 (v2) is constructed in the xy-plane
        x2 = v1_dot_v2 / v1_norm
        
        # Clamp the argument of sqrt to avoid domain errors from floating-point inaccuracies
        y2_sq = v2_sq - x2**2
        if np.isclose(y2_sq, 0):
             y2_sq = 0
        elif y2_sq < 0:
            raise ValueError(f"Invalid metric tensor; sqrt of negative number ({y2_sq}) for y2 component.")
            
        y2 = np.sqrt(y2_sq)
        gen_2 = np.array([x2, y2, 0.0])
        
        # Ensure v1 and v2 are not collinear
        if np.isclose(y2, 0):
             raise ValueError("Generators v1 and v2 are collinear, cannot form a 3D basis.")

        # Generator 3 (v3) is constructed in 3D space
        x3 = v1_dot_v3 / v1_norm
        y3 = (v2_dot_v3 - x2 * x3) / y2
        
        # Clamp the argument of sqrt for the z-component
        z3_sq = v3_sq - x3**2 - y3**2
        if np.isclose(z3_sq, 0):
            z3_sq = 0
        elif z3_sq < 0:
            raise ValueError(f"Invalid metric tensor; sqrt of negative number ({z3_sq}) for z3 component.")

        z3 = np.sqrt(z3_sq)
        gen_3 = np.array([x3, y3, z3])

        return np.array([gen_1, gen_2, gen_3])

    def to_generators_max(self, lattice_step_size: float):
        v0_dot_v1, v0_dot_v2, _, v1_dot_v2, _, _ = self.conorms.conorms * lattice_step_size

        v0_norm = np.sqrt(self[0] * lattice_step_size)
        v1_norm = np.sqrt(self[1] * lattice_step_size)
        v2_norm = np.sqrt(self[2] * lattice_step_size)

        # x = v0_dot_v1 / (lattice_step_size ** 2 * v0_norm * v1_norm) # Max
        x = v0_dot_v1 / (v0_norm * v1_norm) # David

        y = np.sqrt(1 - x ** 2) # Max + David

        # a = v0_dot_v2 / (lattice_step_size ** 2 * v0_norm * v2_norm) # Max
        a = v0_dot_v2 / (v0_norm * v2_norm) # David
        
        # b = (1 / y) * (v1_dot_v2 / (lattice_step_size ** 2 * v1_norm * v2_norm) - x * a) # Max
        b = (1 / y) * v1_dot_v2 / (v1_norm * v2_norm) - x * a # David

        c = np.sqrt(1 - a ** 2 - b ** 2)

        # v0 = lattice_step_size * v0_norm * np.array([1, 0, 0]) # Max
        v0 = v0_norm * np.array([1, 0, 0]) # David
        # v1 = lattice_step_size * v1_norm * np.array([x, y, 0]) # Max
        v1 = v1_norm * np.array([x, y, 0]) # David
        v2 = v2_norm * np.array([a, b, c]) # Max
        return np.array([v0, v1, v2])

    def to_generators_david(self, lattice_step_size: float):
        v0_dot_v1, v0_dot_v2, _, v1_dot_v2, _, _ = tuple(VONORM_TO_DOT_PRODUCTS @ self.vonorms[:6])
        # v0_dot_v1, v0_dot_v2, _, v1_dot_v2, _, _ = self.conorms # * lattice_step_size
        print(v0_dot_v1 / 2, self.conorms[0] * lattice_step_size)
        # assert v0_dot_v1 == self.conorms[0] / 2

        v0_norm = np.sqrt(self[0] * lattice_step_size)
        v1_norm = np.sqrt(self[1] * lattice_step_size)
        v2_norm = np.sqrt(self[2] * lattice_step_size)

        # x = v0_dot_v1 / (lattice_step_size ** 2 * v0_norm * v1_norm) # Max
        x = v0_dot_v1 * lattice_step_size / (2 * v0_norm * v1_norm) # David

        y = np.sqrt(1 - x ** 2) # Max + David

        # a = v0_dot_v2 / (lattice_step_size ** 2 * v0_norm * v2_norm) # Max
        a = v0_dot_v2 * lattice_step_size / (2 * v0_norm * v2_norm) # David
        
        # b = (1 / y) * (v1_dot_v2 / (lattice_step_size ** 2 * v1_norm * v2_norm) - x * a) # Max
        b = (1 / y) * v1_dot_v2 * lattice_step_size / (2 * v1_norm * v2_norm) - x * a # David

        c = np.sqrt(1 - a ** 2 - b ** 2)

        # v0 = lattice_step_size * v0_norm * np.array([1, 0, 0]) # Max
        v0 = v0_norm * np.array([1, 0, 0]) # David
        # v1 = lattice_step_size * v1_norm * np.array([x, y, 0]) # Max
        v1 = v1_norm * np.array([x, y, 0]) # David
        v2 = v2_norm * np.array([a, b, c]) # Max
        return np.array([v0, v1, v2])
    
    def to_superbasis(self, lattice_step_size: float = 1.0):
        from ..superbasis import Superbasis
        return Superbasis.from_generating_vecs(self.recover_generators(lattice_step_size=lattice_step_size))

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