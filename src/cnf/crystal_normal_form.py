import numpy as np
from pymatgen.core import Lattice, Structure # Or other library
from .lattice import VonormList, Superbasis
from .lattice.permutations import VonormPermutation
from .motif.atomic_motif import FractionalMotif
from .lattice.utils import selling_reduce
from .lattice.rounding import DiscretizedVonormComputer
from .lattice.lattice_normal_form import LatticeNormalForm
from .motif.basis_normal_form import BasisNormalForm
from .unit_cell import UnitCell

class CrystalNormalForm:

    @classmethod
    def from_unit_cell(cls,
                       unit_cell: UnitCell,
                       lattice_step_size: float,
                       motif_step_size: int):
        return cls.from_motif_and_superbasis(unit_cell.motif, unit_cell.superbasis, lattice_step_size, motif_step_size)

    @classmethod
    def from_motif_and_superbasis(cls,
                                  motif: FractionalMotif,
                                  superbasis: Superbasis,
                                  lattice_step_size: float,
                                  motif_step_size: int):
        
        lnf, selling_transform, stabilizer_neighbor_permutations = LatticeNormalForm.from_superbasis(superbasis,
                                                                                                     lattice_step_size,
                                                                                                     return_transforms=True)
        
        motif = motif.apply_unimodular(selling_transform)
    

        bnfs: list[BasisNormalForm] = []
        for stabilizer_permutation in stabilizer_neighbor_permutations:
            unimodular_transform = stabilizer_permutation.to_unimodular_matrix()
            transformed_motif = motif.apply_unimodular(unimodular_transform)
            bnf = BasisNormalForm.from_motif(transformed_motif, motif_step_size)
            bnfs.append(bnf)
        
        sorted_bnfs = sorted(bnfs, key=lambda bnf: bnf.coord_list, reverse=True)
        canonical_bnf = sorted_bnfs[0]
        # print(canonical_bnf)
        return cls(lnf, canonical_bnf, lattice_step_size, motif_step_size)
    
    @classmethod
    def from_motif_and_basis_vecs(cls,
                                  motif: FractionalMotif,
                                  basis_vecs: np.array,
                                  lattice_step_size=1.5,
                                  motif_step=30):
        superbasis = Superbasis.from_generating_vecs(basis_vecs)
        return cls.from_motif_and_superbasis(motif, superbasis, lattice_step_size, motif_step)

    @classmethod
    def from_pymatgen_structure(cls, struct: Structure, lattice_step_size=1.5, motif_step=30):
        motif = FractionalMotif.from_pymatgen_structure(struct)
        superbasis = Superbasis.from_pymatgen_structure(struct)
        return cls.from_motif_and_superbasis(motif, superbasis, lattice_step_size, motif_step)
    
    def __init__(self,
                 lattice_normal_form: LatticeNormalForm,
                 basis_normal_form: BasisNormalForm,
                 xi: float,
                 delta: int):
        """
        Represents a crystal structure in its unique, discretized normal form.

        Args:
            canonical_vonorms (tuple): The 7-integer tuple for the lattice.
            canonical_basis (tuple): The 3(n-1) integer tuple for the basis.
            canonical_permutation (tuple): The 4-integer tuple for the canonical permutation.
            xi (float): The lattice discretization parameter (e.g., in Å²).
            delta (int): The basis discretization parameter (number of divisions).
        """
        self.lattice_normal_form = lattice_normal_form
        self.basis_normal_form = basis_normal_form
        self.xi = xi
        self.delta = delta
    
    @property
    def coords(self):
        lnf_coords = self.lattice_normal_form.coords
        bnf_coords = self.basis_normal_form.coord_list
        return tuple(lnf_coords) + tuple(bnf_coords)

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
        return (f"CrystalNormalForm(lattice={self.lattice_normal_form}, motif={self.basis_normal_form}, "
                f"xi={self.xi}, delta={self.delta})")
    
    def __eq__(self, other: 'CrystalNormalForm'):
        return self.coords == other.coords
    
    def __hash__(self):
        return self.coords.__hash__()