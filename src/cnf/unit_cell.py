from pymatgen.core.structure import Structure

from .lattice import Superbasis
from .motif import FractionalMotif
from .sublattice.gamma_matrices import GammaMatrixGroup
from .sublattice.generation import transform_lattice_vecs, MotifTranslationSet

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
        gamma_matrices = GammaMatrixGroup.for_index(index)
        supercells = []
        for gm in gamma_matrices.matrices:
            new_lattice = transform_lattice_vecs(self.superbasis.generating_vecs(), gm)
            translators = MotifTranslationSet.from_gamma_matrix(gm)
            new_motif = translators.apply_to_motif(self.motif)
            supercells.append(UnitCell(Superbasis.from_generating_vecs(new_lattice), new_motif))
        return supercells

