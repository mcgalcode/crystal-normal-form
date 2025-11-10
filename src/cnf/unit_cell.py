from pymatgen.core.structure import Structure
import numpy as np
from .linalg import MatrixTuple
from .lattice import Superbasis
from .motif import FractionalMotif
from .sublattice.gamma_matrices import GammaMatrixGroup
from .sublattice.sublattice_generator import SublatticeGenerator
from cnf.lattice.selling import SuperbasisSellingReducer
from cnf.lattice.permutations import VonormPermutation

from .cnf_constructor import CNFConstructor
from .crystal_normal_form import CrystalNormalForm

class UnitCell():

    @classmethod
    def from_pymatgen_structure(cls, struct: Structure):
        superbasis = Superbasis.from_pymatgen_structure(struct)
        motif = FractionalMotif.from_pymatgen_structure(struct)
        return cls(superbasis, motif)

    @classmethod
    def from_cnf(cls, cnf: CrystalNormalForm):
        return cls.from_pymatgen_structure(cnf.reconstruct())
    
    @classmethod
    def from_cif(cls, cif_path: str):
        return cls.from_pymatgen_structure(Structure.from_file(cif_path))

    def __init__(self, superbasis: Superbasis, motif: FractionalMotif):
        self.superbasis = superbasis
        self.motif = motif
    
    def supercells(self, index: int):
        sg = SublatticeGenerator.for_index(index)
        lattice_vec_sets = sg.generate_sublattice_vector_sets(self.superbasis.generating_vecs())
        motifs = sg.generate_sublattice_motifs(self.motif)
        return [UnitCell(Superbasis.from_generating_vecs(lvs), motif) for lvs, motif in zip(lattice_vec_sets, motifs)]
    
    def reduce(self):
        reducer = SuperbasisSellingReducer()
        result = reducer.reduce(self.superbasis)
        reduced_sb = result.reduced_object
        reduced_motif = self.motif.apply_unimodular(result.transform_matrix)        
        return UnitCell(reduced_sb, reduced_motif)
    
    def apply_unimodular(self, u: MatrixTuple):
        new_sb = self.superbasis.apply_matrix_transform(u.matrix)
        new_motif = self.motif.apply_unimodular(u)
        return UnitCell(new_sb, new_motif)

    def to_pymatgen_structure(self):
        lattice_vecs = self.superbasis.generating_vecs()
        return Structure(lattice_vecs, self.motif.atoms, self.motif.positions)  

    @property
    def volume(self):
        return self.to_pymatgen_structure().volume
    
    @property
    def vonorms(self):
        return self.superbasis.compute_vonorms()
    
    @property
    def conorms(self):
        return self.vonorms.conorms
    
    @property
    def voronoi_class(self):
        return self.conorms.form.voronoi_class

    def to_cnf(self, xi, delta, verbose=False):
        c = CNFConstructor(xi, delta, verbose_logging=verbose)
        res = c.from_motif_and_superbasis(self.motif, self.superbasis)
        return res.cnf
    
    def is_obtuse(self, tol=1e-8):
        return self.superbasis.is_obtuse(tol=tol)
    
    def to_cif(self, fpath):
        if ".cif" not in fpath:
            raise ValueError("Must provide CIF filepath ending in .cif!")
        self.to_pymatgen_structure().to_file(fpath)

    def __len__(self):
        return len(self.motif)