import numpy as np
from pymatgen.core import Lattice, Structure # Or other library
from .lattice import VonormList, Superbasis
from .lattice.permutations import VonormPermutation
from .motif.atomic_motif import FractionalMotif
from .lattice.utils import selling_reduce
from .lattice.rounding import DiscretizedVonormComputer
from .motif.basis_normal_form import BasisNormalForm

class CrystalNormalForm:


    @classmethod
    def from_motif_and_superbasis(cls,
                                  motif: FractionalMotif,
                                  superbasis: Superbasis,
                                  lattice_step_size: float,
                                  motif_step_size: int):
        vonorms = superbasis.compute_vonorms()
        vonorms = VonormList(DiscretizedVonormComputer(vonorms.vonorms, lattice_step_size).find_closest_valid_vonorms())
        reduced_vonorms, _, transform_mat = selling_reduce(vonorms, return_transform_mat=True)
        motif = motif.apply_unimodular(transform_mat)
        conorms = reduced_vonorms.conorms

        permissible_permutations = conorms.permissible_permutations
        permuted_vonorm_lists: list[tuple[VonormList, VonormPermutation]] = []
        for conorm_permutation in permissible_permutations:
            vonorm_permutation = conorm_permutation.to_vonorm_permutation()
            permuted_vlist = reduced_vonorms.apply_permutation(vonorm_permutation)
            permuted_vonorm_lists.append((permuted_vlist, vonorm_permutation))

        sorted_vlists = sorted(permuted_vonorm_lists, key=lambda pair: pair[0].vonorms, reverse=False)

        # print(f"Identified permissible vonorm permutations:")
        # for pair in sorted_vlists:
        #     print(f"{pair[1]} -> {pair[0]}")
        
        stabilizer_perms = [pair[1] for pair in permuted_vonorm_lists if pair[0] == sorted_vlists[0][0]]    
        # print("")

        # print(f"Identified stabilizers permutations for maximally ascending {sorted_vlists[0][0]}:")
        # for s in stabilizer_perms:
        #     print(f"{s}")
        
        # print("")

        bnfs: list[BasisNormalForm] = []
        for stabilizer_permutation in stabilizer_perms:
            unimodular_transform = -1*stabilizer_permutation.to_unimodular_matrix()
            transformed_motif = motif.apply_unimodular(unimodular_transform)
            bnf = BasisNormalForm.from_motif(transformed_motif, motif_step_size)
            bnfs.append(bnf)
        
        sorted_bnfs = sorted(bnfs, key=lambda bnf: bnf.coord_list, reverse=True)
        canonical_bnf = sorted_bnfs[0]
        # print(canonical_bnf)
        return cls(sorted_vlists[0][0], canonical_bnf)
    
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
    
    def __init__(self, canonical_vonorms, canonical_basis, xi=1.5, delta=30):
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
                f"xi={self.xi}, delta={self.delta})")