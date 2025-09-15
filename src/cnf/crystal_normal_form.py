import numpy as np
from pymatgen.core import Lattice, Structure # Or other library
from .lattice import VonormList, Superbasis
from .lattice.atomic_motif import AtomicMotif
from .lattice.utils import selling_reduce
from .lattice.sorting import sort_vonorms
from .lattice.stabilizer import search_for_stabilizers
from .lattice_normal_form.rounding import DiscretizedVonormComputer
from .basis_normal_form import BasisNormalForm

class CrystalNormalForm:

    @classmethod
    def from_structure(cls, struct: Structure, lattice_step_size=1.5):
        motif = AtomicMotif.from_structure(struct)
        superbasis = Superbasis.from_pymatgen_structure(struct)
        vonorms = superbasis.compute_vonorms()
        vonorms = VonormList(DiscretizedVonormComputer(vonorms.vonorms, lattice_step_size).find_closest_valid_vonorms())
        vonorms, transform_mat = selling_reduce(vonorms, return_transform_mat=True)
        motif = motif.transform(transform_mat)
        _, sort_transform_mat = sort_vonorms(vonorms, return_transform_mat=True)
        motif = motif.transform(sort_transform_mat)

        stabilizers, _ = search_for_stabilizers(vonorms)
        bnfs = []
        for stabilizer, transform in stabilizers.items():
            transformed_motif = motif.transform(transform)
            bnf = BasisNormalForm.from_motif(transformed_motif)
            bnfs.append(bnf)
        # bnfs.sort() ???

        
    
    def __init__(self, canonical_vonorms, canonical_basis, canonical_permutation, xi=1.5, delta=30):
        """
        Represents a crystal structure in its unique, discretized normal form.

        Args:
            canonical_vonorms (tuple): The 7-integer tuple for the lattice.
            canonical_basis (tuple): The 3(n-1) integer tuple for the basis.
            canonical_permutation (tuple): The 4-integer tuple for the canonical permutation.
            xi (float): The lattice discretization parameter (e.g., in Å²).
            delta (int): The basis discretization parameter (number of divisions).
        """
        self.vonorms = canonical_vonorms
        self.basis = canonical_basis
        self.permutation = canonical_permutation
        self.xi = xi
        self.delta = delta

    def reconstruct(self) -> 'Structure':
        """
        Reconstructs the full crystal structure (lattice and atomic sites) from
        the stored normal form.
        """
        # This method will use self.xi to convert integer vonorms back to floats,
        # run the `vonorm -> superbasis` algorithm, use self.permutation to
        # correctly label the basis, and then use self.delta to place the atoms.
        pass

    def __repr__(self):
        return (f"CrystalNormalForm(vonorms={self.vonorms}, basis={self.basis}, "
                f"perm={self.permutation}, xi={self.xi}, delta={self.delta})")