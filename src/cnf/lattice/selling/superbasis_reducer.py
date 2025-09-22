import numpy as np

from .selling_reducer import SellingReducer
from ..superbasis import Superbasis

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



class SuperbasisSellingReducer(SellingReducer):

    def get_dot_product_for_pair(self, obj: Superbasis, pair: tuple[int, int]):
        i, j = pair
        return np.dot(obj.superbasis_vecs[i], obj.superbasis_vecs[j])
    
    def get_transformed_object(self, superbasis: Superbasis, pair):
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
        i, j = pair
        vecs = superbasis.superbasis_vecs
        transformed_vecs = np.zeros(vecs.shape)

        # First assign the easy ones:
        transformed_vecs[i] = -vecs[i]
        transformed_vecs[j] = vecs[j]

        # Now do the other two. The labeling (k, l in Kurlin) is arbitrary
        # and they are both transformed the same way: by adding v_i to their
        # previous values
        other_idxs = {0, 1, 2, 3} - {i, j}
        for other_idx in other_idxs:
            transformed_vecs[other_idx] = vecs[other_idx] + vecs[i]        
        return Superbasis(transformed_vecs)

