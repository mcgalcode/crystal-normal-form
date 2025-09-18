import numpy as np

from .gamma_matrices import GammaMatrixTuple
from .generation import transform_basis_position

from ..motif import FractionalMotif

class MotifTranslationSet():

    @classmethod
    def from_gamma_matrix(cls, gamma_mat: GammaMatrixTuple):
        gamma_inv = gamma_mat.inverse()
        gamma_11, gamma_22, gamma_33 = tuple(np.diag(gamma_mat.matrix))
        translation_vecs = []
        for z1 in range(gamma_11):
            for z2 in range(gamma_22):
                for z3 in range(gamma_33):
                    if z1 == 0 and z2 == 0 and z3 == 0:
                        continue
                    
                    z_vec = np.array([z1, z2, z3])
                    transformed = np.mod(gamma_inv @ z_vec, 1)
                    translation_vecs.append(transformed)
        return MotifTranslationSet(translation_vecs, gamma_mat)

    def __init__(self, vecs: list[np.ndarray], generating_matrix: GammaMatrixTuple):
        self.vecs = vecs
        self.generating_matrix = generating_matrix
    
    def __len__(self):
        return len(self.vecs)
    
    def apply_to_coord(self, pos: np.ndarray):
        images = []
        transformed = transform_basis_position(pos, self.generating_matrix)
        images.append(transformed)
        for v in self.vecs:
            images.append(transformed + v)
        return images
    
    def apply_to_motif(self, motif: FractionalMotif):
        
        if not isinstance(motif, FractionalMotif):
            raise ValueError(f"Tried to apply motif translation to something other than FractionalMotif instance: {motif}")
        
        new_el_pos_map = {}

        for el in motif.unique_elements():
            new_el_pos_map[el] = []
            for pos in motif.get_element_positions(el):
                position_images = self.apply_to_coord(np.asarray(pos))
                new_el_pos_map[el].extend(position_images)
        
        return FractionalMotif(new_el_pos_map)