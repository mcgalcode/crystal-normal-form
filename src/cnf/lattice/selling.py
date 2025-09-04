import numpy as np

CANONICAL_PAIRS = [
    (0,1),
    (0,2),
    (0,3),
    (1,2),
    (1,3),
    (2,3),
]

LABELS_TO_COLS = {
    1: np.array([1, 0, 0]),
    2: np.array([0, 1, 0]),
    3: np.array([0, 0, 1]),
    0: np.array([-1, -1, -1]),
}

def build_unimodular_selling_transform_matrix(selling_pair: tuple):
    columns = []
    i, j = selling_pair
    # We only care about vectors 1,2, and 3
    for lattice_vec_label in range(1,4):
        if lattice_vec_label == i:
            col = - LABELS_TO_COLS[lattice_vec_label]
        elif lattice_vec_label == j:
            col = LABELS_TO_COLS[lattice_vec_label]
        else:
            col = LABELS_TO_COLS[lattice_vec_label] + LABELS_TO_COLS[i]
        columns.append(col)
    return np.array(columns).T

SELLING_TRANSFORM_MATRICES = {}
SELLING_TRANSFORM_INVERSE_MATRICES = {}

for p in CANONICAL_PAIRS:
    SELLING_TRANSFORM_MATRICES[p] = build_unimodular_selling_transform_matrix(p)

for p in CANONICAL_PAIRS:
    SELLING_TRANSFORM_INVERSE_MATRICES[p] = np.linalg.inv(SELLING_TRANSFORM_MATRICES[p]).astype(int)

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
                return (i, j)
    return None


def get_v0_from_generating_vecs(generating_vecs):
    return sum([-np.array(v) for v in generating_vecs])

def get_obtuse_superbasis(lattice_generating_vecs: np.array):
    """Given any basis for a lattice, returns the unique set of obtuse superbasis
    vectors. Note that these vectors are not assigned any particular labels,
    so producing the vonorms will first require you to be intentional about assigning
    indices to these vectors.

    Parameters
    ----------
    lattice_generating_vecs : np.array
        A numpy array where each ROW is a lattice basis vector
    """
    # Use the transpose because we expect the basis vecs to come in 
    # the standard form (row vectors)
    current_basis_vecs = np.copy(lattice_generating_vecs)

    # First, compute the fourth vector of the superbasis
    # This is just the sum of the others.
    v0 = get_v0_from_generating_vecs(current_basis_vecs)
    current_basis_vecs = np.array([v0, *current_basis_vecs])
    acute_pair = find_first_acute_pair(current_basis_vecs)

    MAX_NUM_ATTEMPTS = 100
    num_attempts = 0
    # NEXT: We actually have to look through all pairs (including v0) to find the acute vector pairs...
    while acute_pair is not None and num_attempts < MAX_NUM_ATTEMPTS:
        # We find the first pair that is currently acute
        first_acute_idx, second_acute_idx = acute_pair
        current_basis_vecs = apply_selling_transformation(
            current_basis_vecs,
            first_acute_idx,
            second_acute_idx
        )

        acute_pair = find_first_acute_pair(current_basis_vecs)
        num_attempts += 1
        # print(current_basis_vecs)

    return current_basis_vecs

def apply_selling_transformation(superbasis_vectors: np.array,
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
    # print(f"Appling with {(i, j)}")


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