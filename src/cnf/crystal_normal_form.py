from pymatgen.core import Structure # Or other library
from .lattice.lattice_normal_form import LatticeNormalForm
from .motif.basis_normal_form import BasisNormalForm

class CrystalNormalForm:
    
    def __init__(self,
                 lattice_normal_form: LatticeNormalForm,
                 basis_normal_form: BasisNormalForm):
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
        self.xi = lattice_normal_form.lattice_step_size
        self.delta = basis_normal_form.delta
    
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
        return self.coords == other.coords and self.xi == other.xi and self.delta == other.delta
    
    def __hash__(self):
        return (self.coords + tuple([self.xi, self.delta])).__hash__()