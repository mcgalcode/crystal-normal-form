import numpy as np
from pymatgen.core.structure import Structure
from itertools import permutations

VONORM_TO_DOT_PRODUCTS = np.array([
    [-1, -1, 0, 0, 1, 0],
    [-1, 0, -1, 0, 0, 1],
    [0, 1, 1, 0, -1, -1],
    [1, 0, 0, 1, -1, -1],
    [0, -1, 0, -1, 0, 1],
    [0, 0, -1, -1, 1, 0]
])

class LatticeNormalForm():

    @classmethod
    def get_obtuse_superbasis(cls, lattice_generating_vecs: np.array):
        """Given any basis for a lattice, returns the unique set of obtuse superbasis
        vectors. Note that these vectors are not assigned any particular labels,
        so producing the vonorms will first require you to be intentional about assigning
        indices to these vectors.

        #### IMPORTANT:
        # In the Kurlin paper ("An isometry..."), section 4 and Lemmas 4.1-4.5 show that
        the number of obtuse superbases depends on the "voronoi classification" which
        is just a measure of symmetry of the lattice. In other words, this selling reduction
        won't always land on a unique superbasis for high symmetry lattices. Is this
        a problem?

        Maybe? If so, Kurlin gives the method for finding all of the other superbases
        for each of the Voronoi classifications, so once we find the first one, we can
        compute the others, then iterate through all of their vonorm permutations...

        Okay that's actually not right. Let's use the Root Invariant the Kurlin defines.

        Parameters
        ----------
        lattice_generating_vecs : np.array
            A numpy array where each COLUMN is a lattice basis vector
        """
        # Use the transpose because we expect the basis vecs to come in 
        # the standard form (column vectors)
        current_basis_vecs = np.copy(lattice_generating_vecs.T)

        # First, compute the fourth vector of the superbasis
        # This is just the sum of the others.
        v0 = cls.get_v0_from_generating_vecs(current_basis_vecs)
        current_basis_vecs = np.array([v0, *current_basis_vecs])
        print(current_basis_vecs)
        acute_pair = cls.find_first_acute_pair(current_basis_vecs)

        MAX_NUM_ATTEMPTS = 100
        num_attempts = 0
        # NEXT: We actually have to look through all pairs (including v0) to find the acute vector pairs...
        while acute_pair is not None and num_attempts < MAX_NUM_ATTEMPTS:
            # We find the first pair that is currently acute
            first_acute_idx, second_acute_idx = acute_pair
            print("APPLYING: ", acute_pair)
            current_basis_vecs = cls.apply_selling_transformation(
                current_basis_vecs,
                first_acute_idx,
                second_acute_idx
            )

            acute_pair = cls.find_first_acute_pair(current_basis_vecs)
            num_attempts += 1
            print(current_basis_vecs)

        return current_basis_vecs
    
    @classmethod
    def apply_selling_transformation(cls,
                                     superbasis_vectors: np.array,
                                     first_acute_idx: int,
                                     second_acute_idx: int) -> np.array:
        """Performs the Selling reduction step described in Lemma A.1 (reduction)
        in Kurlin, 2022 "A complete isometry classification of 3-dimensional
        lattices". You can only apply this function after you have identified a pair
        of vectors in your superbasis whose dot product is positive (i.e. there is
        an acute angle between them).

        Note: This transformation is also defined in David Mrdjenovich's thesis
        in Proof. [12] on page #18. It shows the matrix form of the reduction step
        update as well.

        Parameters
        ----------
        superbasis_vectors : np.array
            The four non-obtuse superbasis vectors in need of reduction (v_i..l in
            the paper)
        first_acute_idx : int
            The index of the first vector in the pair with a positive dot product.
        second_acute_idx : int
            The index of the second vector in the pair with a positive dot product.

        Returns
        -------
        np.array
            A new matrix where the rows are the new superbasis vectors
            after the reduction step has been applied one time.
        """
        transformed_vecs = np.zeros(superbasis_vectors.shape)
        i = first_acute_idx
        j = second_acute_idx

        # First assign the easy ones:
        transformed_vecs[i] = -superbasis_vectors[i]
        transformed_vecs[j] = superbasis_vectors[j]

        # Now do the other two. The labeling (k, l in Kurlin) is arbitrary
        # and they are both transformed the same way: by adding v_i to their
        # previous values
        other_idxs = {0, 1, 2, 3} - {i, j}
        for other_idx in other_idxs:
            transformed_vecs[other_idx] = superbasis_vectors[other_idx] + superbasis_vectors[i]
        
        return transformed_vecs


    @staticmethod
    def get_v0_from_generating_vecs(generating_vecs):
        return sum([-np.array(v) for v in generating_vecs])

    @staticmethod
    def find_first_acute_pair(vecs: np.array):
        """Given a list of vectors (perhaps an obtuse superbasis), iterates through
        the pairwise dot products and if a positive dot product is encountered (a
        pair of vectors separated by an acute angle), the indices of those vectors
        are returned as a tuple, otherwise None is returned.

        Parameters
        ----------
        vecs : np.array
            The vectors to search for positive dot products between

        Returns
        -------
        Union[None | Tuple[int, int]]
            None if all angles are obtuses, or the pair of indices of vectors
            which have an acute angle between them.
        """
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if np.dot(vecs[i], vecs[j]) > 1e-7: # Use a tolerance
                    print(np.dot(vecs[i], vecs[j]))
                    return (i, j)
        return None

    @staticmethod
    def compute_vonorms(superbasis_vectors: np.array) -> tuple[float]:
        """Given a (hopefully) obtuse superbasis, computes the 7
        vonorms associated with them. Labeling/order of the basis vectors
        is important, since it determines the order of the vonorms as well. 
        This function expects the basis vectors to be the ROWS (first index)
        of the supplied array.

        Parameters
        ----------
        superbasis_vectors : np.array
            A list of superbasis vectors (i.e. rows in a matrix)

        Returns
        -------
        tuple[float]
            The list of vonorm values (see pp 25 of DM thesis)
        """
        # rename this for brevity
        sb = superbasis_vectors
        return (
            np.dot(sb[0], sb[0]),
            np.dot(sb[1], sb[1]),
            np.dot(sb[2], sb[2]),
            np.dot(sb[3], sb[3]),
            np.dot(sb[0] + sb[1], sb[0] + sb[1]),
            np.dot(sb[0] + sb[2], sb[0] + sb[2]),
            np.dot(sb[0] + sb[3], sb[0] + sb[3]),
        )
    
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
            permuted_basis = [superbasis_vectors[idx] for idx in p]

            # 3. And compute the vonorms for each basis
            vonorm_list = LatticeNormalForm.compute_vonorms(permuted_basis)
            # print(vonorm_list)
            # 3a. Be sure to round these here
            # this should probably be dtype=int
            rounded_vonorm_list = np.round(np.array(vonorm_list) / epsilon, 0).tolist()
            print(rounded_vonorm_list)
            permutation_vonorm_pairs.append((rounded_vonorm_list, p))

        # 4. Sort in lexicographical (maximally ascending) order
        # By default, python sorts tuples using lexicographical order, and the
        # vonorm lists are already tuples at this point
        sorted_pairs = sorted(permutation_vonorm_pairs, key=lambda x: x[0])

        # See pp 45 of DM thesis: maximally ascending means the values stay
        # low the farthest into the list
        canonical_vonorms = sorted_pairs[0][0]

        matching_permutations = [p[1] for p in sorted_pairs if np.all(np.isclose(p[0], canonical_vonorms))]
        return (matching_permutations, canonical_vonorms)
    
    def canonical_vonorm_to_canonical_generators(canonical_vonorm_string, discretization: float):
        v0_dot_v1, v0_dot_v2, v0_dot_v3, v1_dot_v2, v1_dot_v3, v2_dot_v3 = LatticeNormalForm.get_dot_products(np.array(canonical_vonorm_string) * discretization)

        v0_norm = np.sqrt(canonical_vonorm_string[0])
        v1_norm = np.sqrt(canonical_vonorm_string[1])
        v2_norm = np.sqrt(canonical_vonorm_string[2])

        x = v0_dot_v1 / (discretization ** 2 * v0_norm * v1_norm)
        y = np.sqrt(1 - x ** 2)

        a = v0_dot_v2 / (discretization ** 2 * v0_norm * v2_norm)
        b = (1 / y) * (v1_dot_v2 / (discretization ** 2 * v1_norm * v2_norm) - x * a)
        c = np.sqrt(1 - a ** 2 - b ** 2)

        v0 = discretization * v0_norm * np.array([1, 0, 0])
        v1 = discretization * v1_norm * np.array([x, y, 0])
        v2 = discretization * v2_norm * np.array([a, b, c])
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
    

    def to_crystal_normal_form(structure: Structure, xi=1.5):
        lattice_matrix = structure.lattice.matrix
        superbasis = LatticeNormalForm.get_obtuse_superbasis(lattice_matrix.T)
        matching_permutations, canonical_vonorm_list = LatticeNormalForm.get_canonicalized_superbasis_and_vonorms(superbasis, xi)


# class AtomicBasisNormalForm():

#     @staticmethod
#     def find_canonical_ordering(canonical_vonorm_)