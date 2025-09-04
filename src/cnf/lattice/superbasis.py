import numpy as np

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from .vonorm_list import VonormList
from .selling import find_first_acute_pair, apply_selling_transformation

class Superbasis():

    @classmethod
    def from_pymatgen_lattice(cls, lattice: Lattice):
        lattice_vecs = lattice.matrix
        v0 = - lattice_vecs[0] - lattice_vecs[1] - lattice_vecs[2]
        superbasis_vecs = np.array([v0, *lattice_vecs])
        return cls(superbasis_vecs)

    @classmethod
    def from_pymatgen_structure(cls, struct: Structure):
        return cls.from_pymatgen_lattice(struct.lattice)

    def __init__(self, lattice_vecs: np.array):
        self.lattice_vecs = lattice_vecs

    def generating_vecs(self):
        return self.lattice_vecs[1:]
    
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
        superbasis_vectors : np.array
            A list of superbasis vectors (i.e. rows in a matrix)

        Returns
        -------
        VonormList
            The list of vonorm values (see pp 25 of DM thesis)
        """
        # rename this for brevity
        lv = self.lattice_vecs
        return VonormList((
            np.dot(lv[0], lv[0]),
            np.dot(lv[1], lv[1]),
            np.dot(lv[2], lv[2]),
            np.dot(lv[3], lv[3]),
            np.dot(lv[0] + lv[1], lv[0] + lv[1]),
            np.dot(lv[0] + lv[2], lv[0] + lv[2]),
            np.dot(lv[0] + lv[3], lv[0] + lv[3]),
        ))

    def selling_transform(self) -> tuple["Superbasis", tuple[int, int]]:
        acute_pair = find_first_acute_pair(self.lattice_vecs)
        if acute_pair is None:
            return self
        first_acute_idx, second_acute_idx = acute_pair
        new_basis_vecs = apply_selling_transformation(
            self.lattice_vecs,
            first_acute_idx,
            second_acute_idx
        )
        return Superbasis(new_basis_vecs), acute_pair
    
    def __eq__(self, other: "Superbasis"):
        return np.all(np.isclose(self.lattice_vecs, other.lattice_vecs))