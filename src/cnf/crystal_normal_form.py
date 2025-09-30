import numpy as np
from pymatgen.core import Structure # Or other library
from .lattice import Superbasis
from .motif.atomic_motif import FractionalMotif
from .lattice.lnf_constructor import LatticeNormalFormConstructor
from .lattice.lattice_normal_form import LatticeNormalForm
from .motif.basis_normal_form import BasisNormalForm
from .unit_cell import UnitCell

class CrystalNormalForm:

    @classmethod
    def from_unit_cell(cls,
                       unit_cell: UnitCell,
                       lattice_step_size: float,
                       motif_step_size: int,
                       verbose_logging = False):
        return cls.from_motif_and_superbasis(
            unit_cell.motif,
            unit_cell.superbasis,
            lattice_step_size,
            motif_step_size,
            verbose_logging=verbose_logging
        )

    @classmethod
    def from_motif_and_superbasis(cls,
                                  motif: FractionalMotif,
                                  superbasis: Superbasis,
                                  lattice_step_size: float,
                                  motif_step_size: int,
                                  verbose_logging: bool = False):
        lnf_constructor = LatticeNormalFormConstructor(lattice_step_size, verbose_logging)
        lnf_construction_result = lnf_constructor.build_lnf_from_superbasis(superbasis)

        undisc_result = lnf_construction_result.undiscretized_canonicalization_result
        disc_result = lnf_construction_result.discretized_canonicalization_result
        if verbose_logging:
            print(f"Successfully constructed LNF! {lnf_construction_result.lnf}")
        motif = motif.apply_unimodular(undisc_result.selling_transform_mat)
        
        if verbose_logging:
            print(f"Found {len(disc_result.stabilizer_permutations)} stabilizer permutations...")
        
        init_perm_mat_group = undisc_result.stabilizer_permutations[0]
        bnfs: list[BasisNormalForm] = []
        motif = motif.apply_unimodular(init_perm_mat_group.matrix)
        motif = motif.apply_unimodular(disc_result.selling_transform_mat)

        for stabilizer_perm in disc_result.stabilizer_permutations:
            transformed_motif = motif.apply_unimodular(stabilizer_perm.matrix)
            bnf = BasisNormalForm.from_motif(transformed_motif, motif_step_size)
            bnfs.append(bnf)
    
        
        sorted_bnfs = sorted(bnfs, key=lambda bnf: bnf.coord_list, reverse=True)
        canonical_bnf = sorted_bnfs[0]
        if verbose_logging:
            print(f"Found BNF! {canonical_bnf}")
        return cls(lnf_construction_result.lnf, canonical_bnf, lattice_step_size, motif_step_size)
    
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
        lattice_vecs = self.lattice_normal_form.to_superbasis().generating_vecs()
        motif = self.basis_normal_form.to_motif()
        return Structure(lattice_vecs, motif.atoms, motif.positions)

    def __repr__(self):
        return (f"CrystalNormalForm(lattice={self.lattice_normal_form}, motif={self.basis_normal_form}, "
                f"xi={self.xi}, delta={self.delta})")
    
    def __eq__(self, other: 'CrystalNormalForm'):
        return self.coords == other.coords
    
    def __hash__(self):
        return self.coords.__hash__()