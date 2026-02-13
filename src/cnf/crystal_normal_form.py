from pymatgen.core import Structure # Or other library
from .lattice.lattice_normal_form import LatticeNormalForm
from .motif.motif_normal_form import MotifNormalForm
import json

class CrystalNormalForm:

    @classmethod
    def from_pmg_struct(self, struct: Structure, xi: float, delta: int):
        from .cnf_constructor import CNFConstructor
        return CNFConstructor(xi, delta).from_pymatgen_structure(struct).cnf

    @classmethod
    def from_tuple(cls, tuple, elements, xi, delta):
        lnf_tup = tuple[:7]
        mnf_tup = tuple[7:]
        lnf = LatticeNormalForm.from_coords(lnf_tup, xi)
        mnf = MotifNormalForm(mnf_tup, elements, delta)
        return cls(lnf, mnf)

    def __init__(self,
                 lattice_normal_form: LatticeNormalForm,
                 motif_normal_form: MotifNormalForm):
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
        self.motif_normal_form = motif_normal_form
        self.xi = lattice_normal_form.lattice_step_size
        self.delta = motif_normal_form.delta
        lnf_coords = self.lattice_normal_form.coords
        mnf_coords = self.motif_normal_form.coord_list      
        self.coords = tuple(lnf_coords) + tuple(mnf_coords)
        self._hash_tuple = self.coords + tuple([self.xi, self.delta])
    
    def to_discretized_motif(self):
        return self.motif_normal_form.to_discretized_motif()

    @property
    def motif_coord_matrix(self):
        return self.to_discretized_motif().coord_matrix
    
    @property
    def elements(self):
        return self.motif_normal_form.elements

    def reconstruct(self) -> 'Structure':
        """
        Reconstructs the full crystal structure (lattice and atomic sites) from
        the stored normal form.
        """
        # This method will use self.xi to convert integer vonorms back to floats,
        # run the `vonorm -> superbasis` algorithm, use self.permutation to
        # correctly label the basis, and then use self.delta to place the atoms.
        lattice_vecs = self.lattice_normal_form.to_superbasis().generating_vecs()
        motif = self.motif_normal_form.to_motif()
        return Structure(lattice_vecs, motif.atoms, motif.positions)
    
    @property
    def voronoi_class(self):
        return self.lattice_normal_form.vonorms.conorms.form.voronoi_class

    def __repr__(self):
        return (f"CrystalNormalForm(lattice={self.lattice_normal_form}, motif={self.motif_normal_form}, "
                f"xi={self.xi}, delta={self.delta})")
    
    def __eq__(self, other: 'CrystalNormalForm'):
        return self.coords == other.coords and self.xi == other.xi and self.delta == other.delta
    
    def __hash__(self):
        return self._hash_tuple.__hash__()
    
    @classmethod
    def from_dict(cls, d: dict):
        lnf = LatticeNormalForm.from_dict(d["lnf"])
        mnf = MotifNormalForm.from_dict(d["mnf"])
        return cls(lnf, mnf)
    
    @classmethod
    def from_file(cls, fname: str):
        with open(fname, 'r') as f:
            d = json.load(f)
            return cls.from_dict(d)
        
    def as_str_key(self):
        return f"{self.coords.__repr__()}-{self.elements.__repr__()}-{self.xi}-{self.delta}"
    
    def to_file(self, fname: str):
        with open(fname, 'w') as f:
            json.dump(self.to_dict(), f)
    
    def to_dict(self):
        return {
            "lnf": self.lattice_normal_form.to_dict(),
            "mnf": self.motif_normal_form.to_dict()
        }