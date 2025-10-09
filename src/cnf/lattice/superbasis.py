import numpy as np

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from .permutations import VonormPermutation
from .voronoi.vonorm_list import VonormList
from .vonorm_unimodular import VORONOI_IDX_TO_COLUMNS_TO_SUM

def get_v0_from_generating_vecs(generating_vecs):
    return sum([-np.array(v) for v in generating_vecs])
class Superbasis():

    @classmethod
    def from_pymatgen_lattice(cls, lattice: Lattice):
        return cls.from_generating_vecs(np.array(lattice.matrix))

    @classmethod
    def from_pymatgen_structure(cls, struct: Structure):
        return cls.from_pymatgen_lattice(struct.lattice)
    
    @classmethod
    def from_generating_vecs(cls, generating_vecs: np.ndarray):
        v3 = get_v0_from_generating_vecs(generating_vecs)
        superbasis_vecs = np.array([*generating_vecs, v3])
        return cls(superbasis_vecs)

    def __init__(self, superbasis_vecs: np.ndarray):
        self.superbasis_vecs = superbasis_vecs
    
    def is_superbasis(self, tol=1e-5):
        expected_v3 = get_v0_from_generating_vecs(self.generating_vecs())
        return np.all(np.abs(expected_v3 - self.superbasis_vecs[3]) < tol)
    
    def apply_permutation(self, perm: VonormPermutation):
        new_superbasis_vecs = []
        for idx in perm[:4]:
            col = np.zeros(3)
            for col_idx in VORONOI_IDX_TO_COLUMNS_TO_SUM[idx]:
                col = col + np.array(self.superbasis_vecs[col_idx])
            new_superbasis_vecs.append(col)
        return Superbasis(new_superbasis_vecs)

    
    def generating_vecs(self) -> np.array:
        """Returns the generating vectors of this superbasis as rows in a matrix

        Returns
        -------
        np.array
            A matrix in which each row represents a generating vector
        """
        return self.superbasis_vecs[:3]
    
    def is_obtuse(self, tol=0):
        return self.compute_vonorms().is_obtuse(tol=tol)
    
    def compute_vonorms(self) -> VonormList:
        """Given a (hopefully) obtuse superbasis, computes the 7
        vonorms associated with them. Labeling/order of the basis vectors
        is important, since it determines the order of the vonorms as well. 
        This function expects the basis vectors to be the ROWS (first index)
        of the supplied array.

        Parameters
        ----------
        superbasis_vectors : np.ndarray
            A list of superbasis vectors (i.e. rows in a matrix)

        Returns
        -------
        VonormList
            The list of vonorm values (see pp 25 of DM thesis)
        """
        # rename this for brevity
        lv = self.superbasis_vecs
        return VonormList((
            np.dot(lv[0], lv[0]),
            np.dot(lv[1], lv[1]),
            np.dot(lv[2], lv[2]),
            np.dot(lv[3], lv[3]),
            np.dot(lv[0] + lv[1], lv[0] + lv[1]),
            np.dot(lv[0] + lv[2], lv[0] + lv[2]),
            np.dot(lv[0] + lv[3], lv[0] + lv[3]),
        ))
    
    def apply_matrix_transform(self, mat_transform: np.ndarray):
        new_vecs = self.generating_vecs().T @ mat_transform
        return Superbasis.from_generating_vecs(new_vecs.T)
    
    def v0(self):
        return self.superbasis_vecs[0]
    
    def v1(self):
        return self.superbasis_vecs[1]
    
    def v2(self):
        return self.superbasis_vecs[2]
    
    def v3(self):
        return self.superbasis_vecs[3]
    
    def __eq__(self, other: "Superbasis"):
        return np.all(np.isclose(self.superbasis_vecs, other.superbasis_vecs))
    
    def __repr__(self):
        float_vecs = [[float(i) for i in vec] for vec in self.superbasis_vecs]
        return f"Superbasis({' '.join(['v' + str(idx) + ': ' + str(list(v)) for idx, v in enumerate(float_vecs)])})"