import numpy as np

from .gamma_matrices import GammaMatrixGroup, GammaMatrixTuple
from .motif_translation_set import MotifTranslationSet
from ..motif import FractionalMotif

def transform_lattice_vecs(lvecs: np.ndarray, gmat: GammaMatrixTuple):
    """Converts lattice vecs (as rows) to supercell lattice vecs
    according to the provided gamma matrix

    Parameters
    ----------
    lvecs : np.array
        The lattice vecs for which a supercell should be generated
        using gmat
    """
    return (lvecs.T @ gmat.matrix).T

class SublatticeGenerator():

    @classmethod
    def for_index(cls, N: int):
        gmg = GammaMatrixGroup.for_index(N)
        return cls(gmg)
    
    def __init__(self, sublattice_generating_matrix_group: GammaMatrixGroup):
        self.generating_matrix_group = sublattice_generating_matrix_group

    def generate_sublattice_vector_sets(self,
                                        lattice_vecs: np.ndarray,
                                        return_generating_matrices: bool = False):
        lattice_vec_sets = []
        for gm in self.generating_matrix_group.ordered_matrices:
            lattice_vec_sets.append(transform_lattice_vecs(lattice_vecs, gm))
        
        if return_generating_matrices:
            return lattice_vec_sets, self.generating_matrix_group.ordered_matrices
        else:
            return lattice_vec_sets
    
    def generate_sublattice_motifs(self,
                                   motif: FractionalMotif,
                                   return_generating_matrices: bool = False):
        new_motifs = []
        for gm in self.generating_matrix_group.ordered_matrices:
            translators = MotifTranslationSet.from_gamma_matrix(gm)
            new_motifs.append(translators.apply_to_motif(motif))
        
        if return_generating_matrices:
            return new_motifs, self.generating_matrix_group.ordered_matrices
        else:
            return new_motifs