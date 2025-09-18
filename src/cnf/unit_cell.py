from pymatgen.core.structure import Structure

from .lattice import Superbasis
from .motif import FractionalMotif
from .sublattice.gamma_matrices import GammaMatrixGroup
from .sublattice.sublattice_generator import SublatticeGenerator

class UnitCell():

    @classmethod
    def from_pymatgen_structure(cls, struct: Structure):
        superbasis = Superbasis.from_pymatgen_structure(struct)
        motif = FractionalMotif.from_pymatgen_structure(struct)
        return cls(superbasis, motif)

    def __init__(self, superbasis: Superbasis, motif: FractionalMotif):
        self.superbasis = superbasis
        self.motif = motif
    
    def supercells(self, index: int):
        sg = SublatticeGenerator.for_index(index)
        lattice_vec_sets = sg.generate_sublattice_vector_sets(self.superbasis.generating_vecs())
        motifs = sg.generate_sublattice_motifs(self.motif)
        return [UnitCell(Superbasis.from_generating_vecs(lvs), motif) for lvs, motif in zip(lattice_vec_sets, motifs)]

