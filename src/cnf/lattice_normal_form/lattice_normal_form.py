import numpy as np
from itertools import permutations

from ..lattice import Superbasis
from .rounding import DiscretizedVonormComputer

VONORM_TO_DOT_PRODUCTS = np.array([
    [-1, -1, 0, 0, 1, 0],
    [-1, 0, -1, 0, 0, 1],
    [0, 1, 1, 0, -1, -1],
    [1, 0, 0, 1, -1, -1],
    [0, -1, 0, -1, 0, 1],
    [0, 0, -1, -1, 1, 0]
])

class LatticeNormalForm():



    
    @staticmethod
    def get_canonicalized_superbasis_and_vonorms(superbasis_vectors: np.array, epsilon: float) -> tuple[np.array, list]:
        """Given a list of superbasis vectors (perhaps produced using the methods above),
        identify all of the permutations which yield the maximally ascending list of vonorms,
        and return that list of permutations and the list of maximally ascending vonorms.
        
        Note: it is possible that multiple permutations yield the same list of vonorms.
        Ultimately, the one that is chosen is the one that yields the globally maximally
        ascending list of _atomic basis_ coordinate positions. This is why we return not just
        a single superbasis labeling that yields the canonical vonorm ordering, but a list of
        them.

        Parameters
        ----------
        superbasis_vectors : np.array
            The uncanonicalized vectors (4 rows by 3 columns)
        epsilon : float
            A discretization parameter

        Returns
        -------
        tuple[list[np.array], list]
            A tuple of, first, a list of all of the permutations of the
            original basis vectors which map to the canonical, maximally ascending vonorms,
            and second, a list of those maximally ascending vonorms.
        """
        NUM_SUPERBASIS_VECS = 4

        if len(superbasis_vectors) != NUM_SUPERBASIS_VECS:
            raise ValueError(f"Provided superbasis must have 4 members, but got {len(superbasis_vectors)}")
        
        # 1. Enumerate all the permutations of the superbasis vectors
        # This comes from the S4 group

        vector_idxs = range(NUM_SUPERBASIS_VECS)
        idx_permutations = permutations(vector_idxs, NUM_SUPERBASIS_VECS)

        # 2. Iterate through the permutations
        # This can be optimized because we don't need to compute all the dot
        # products every time. David has rules for how the vonorms change
        # with permutations
        permutation_vonorm_pairs = []
        for p in idx_permutations:
            permuted_basis = Superbasis(np.array([superbasis_vectors[idx] for idx in p]))

            # 3. And compute the vonorms for each basis
            vonorm_list = permuted_basis.compute_vonorms()
            vonorm_computer = DiscretizedVonormComputer(vonorm_list.vonorms, epsilon)
            # 3a. Be sure to round these here
            rounded_vonorm_list = vonorm_computer.find_closest_valid_vonorms()
            # rounded_vonorm_list = (np.array(vonorm_list) / epsilon).tolist()
            permutation_vonorm_pairs.append((rounded_vonorm_list.tolist(), p))

        # 4. Sort in lexicographical (maximally ascending) order
        # By default, python sorts tuples using lexicographical order, and the
        # vonorm lists are already tuples at this point
        sorted_pairs = sorted(permutation_vonorm_pairs, key=lambda x: x[0])

        # See pp 45 of DM thesis: maximally ascending means the values stay
        # low the farthest into the list
        canonical_vonorms = sorted_pairs[0][0]

        matching_permutations = [p[1] for p in sorted_pairs if np.all(np.isclose(p[0], canonical_vonorms))]
        return (matching_permutations, canonical_vonorms)
    
    def canonical_vonorm_to_canonical_generators(canonical_vonorm_string, epsilon: float):
        v0_dot_v1, v0_dot_v2, _, v1_dot_v2, _, _ = LatticeNormalForm.get_dot_products(
            np.array(canonical_vonorm_string) # * epsilon
        )

        v0_norm = np.sqrt(canonical_vonorm_string[0] * epsilon)
        v1_norm = np.sqrt(canonical_vonorm_string[1] * epsilon)
        v2_norm = np.sqrt(canonical_vonorm_string[2] * epsilon)

        # x = v0_dot_v1 / (epsilon ** 2 * v0_norm * v1_norm) # Max
        x = v0_dot_v1 * epsilon / (v0_norm * v1_norm) # David

        y = np.sqrt(1 - x ** 2) # Max + David

        # a = v0_dot_v2 / (epsilon ** 2 * v0_norm * v2_norm) # Max
        a = v0_dot_v2 * epsilon / (v0_norm * v2_norm) # David
        
        # b = (1 / y) * (v1_dot_v2 / (epsilon ** 2 * v1_norm * v2_norm) - x * a) # Max
        b = (1 / y) * (v1_dot_v2 * epsilon / (v1_norm * v2_norm) - x * a) # David

        c = np.sqrt(1 - a ** 2 - b ** 2)

        v0 = v0_norm * np.array([1, 0, 0])
        v1 = v1_norm * np.array([x, y, 0])
        v2 = v2_norm * np.array([a, b, c])
        return np.array([v0, v1, v2])
        
    
    @staticmethod
    def get_dot_products(vonorm_list: np.array):
        """Produces the list of dot products from Theorem 3.0.4 in David's
        thesis:

        v_0 * v_1
        v_0 * v_2
        v_0 * v_3
        v_1 * v_2
        v_1 * v_3
        v_2 * v_3

        Parameters
        ----------
        vonorm_list : np.array
            A list of vonorms (four primary, then three secondary)

        Returns
        -------
        A list of dot products, as described in the docstring.
        """
        return 0.5 * VONORM_TO_DOT_PRODUCTS @ vonorm_list[:6]