import numpy as np
from pymatgen.core import Structure # Or other library
from .lattice import Superbasis
from .motif.atomic_motif import FractionalMotif, DiscretizedMotif
from .lattice.lnf_constructor import LatticeNormalFormConstructor, LatticeNormalFormConstructionResult
from .lattice.voronoi import VonormList
from .motif.bnf_constructor import BNFConstructor, BNFConstructionResult
from .crystal_normal_form import CrystalNormalForm
from .unit_cell import UnitCell

class CNFConstructionResult():

    def __init__(self,
                 lnf_construction_result: LatticeNormalFormConstructionResult,
                 bnf_construction_result: BNFConstructionResult):
        self.lnf_result = lnf_construction_result
        self.bnf_result = bnf_construction_result
    
    @property
    def cnf(self):
        return CrystalNormalForm(self.lnf_result.lnf, self.bnf_result.bnf)

class CNFConstructor():

    def __init__(self,
                 xi: float,
                 delta: int,
                 verbose_logging: bool = False):
        self.xi = xi
        self.delta = delta
        self.verbose_logging = verbose_logging

    def from_unit_cell(self, unit_cell: UnitCell):
        return self.from_motif_and_superbasis(
            unit_cell.motif,
            unit_cell.superbasis
        )

    def from_motif_and_superbasis(self, motif: FractionalMotif, superbasis: Superbasis):
        
        if isinstance(motif, FractionalMotif):
            motif = motif.discretize(self.delta)

        lnf_constructor = LatticeNormalFormConstructor(self.xi, self.verbose_logging)
        lnf_construction_result = lnf_constructor.build_lnf_from_superbasis(superbasis)

        undisc_result = lnf_construction_result.undiscretized_canonicalization_result
        disc_result = lnf_construction_result.discretized_canonicalization_result
        if self.verbose_logging:
            print(f"Successfully constructed LNF! {lnf_construction_result.lnf}")
        
        if self.verbose_logging:
            print(f"Found {len(disc_result.equivalent_transformations)} equivalent transformations...")
        
        pre_transforms = [
            undisc_result.selling_transform_mat,
            undisc_result.equivalent_transformations[0].matrix,
            disc_result.selling_transform_mat,
            disc_result.equivalent_transformations[0].matrix
        ]

        stabilizer = disc_result.canonical_vonorms.stabilizer()

        bnf_constructor = BNFConstructor(pre_transforms, stabilizer)
        bnf_construction_res = bnf_constructor.build(motif)

        if self.verbose_logging:
            print(f"Found BNF! {bnf_construction_res.bnf}")
            
        return CNFConstructionResult(lnf_construction_result, bnf_construction_res)
        
    def from_discretized_vonorms_and_motif(self,
                                           discretized_vonorms: VonormList,
                                           motif: DiscretizedMotif):
        if isinstance(motif, FractionalMotif):
            motif = motif.discretize(self.delta)

        if not discretized_vonorms.is_obtuse():
            raise ValueError(f"Provided discretized Vonorms do not represent obtuse superbasis: {discretized_vonorms.vonorms}")
        
        # Here we follow the same procedure as in #from_motif_and_superbasis but
        # we do not need to perform discretization and we do not need to perform
        # selling reduction
        lnf_constructor = LatticeNormalFormConstructor(self.xi, self.verbose_logging)
        lnf_construction_result = lnf_constructor.build_lnf_from_discretized_vonorms(discretized_vonorms)
        lnf = lnf_construction_result.lnf

        if self.verbose_logging:
            print(f"Successfully constructed LNF! {lnf}")

        pre_transforms = [
            lnf_construction_result.discretized_canonicalization_result.equivalent_transformations[0].matrix
        ]

        stabilizer = lnf_construction_result.discretized_canonicalization_result.canonical_vonorms.stabilizer()

        bnf_constructor = BNFConstructor(pre_transforms, stabilizer)
        bnf_construction_res = bnf_constructor.build(motif)

        if self.verbose_logging:
            print(f"Found BNF! {bnf_construction_res.bnf}")
        
        return CNFConstructionResult(lnf_construction_result, bnf_construction_res)


    def from_motif_and_basis_vecs(self, motif: FractionalMotif, basis_vecs: np.array):
        superbasis = Superbasis.from_generating_vecs(basis_vecs)
        return self.from_motif_and_superbasis(motif, superbasis)

    def from_pymatgen_structure(self, struct: Structure):
        motif = FractionalMotif.from_pymatgen_structure(struct)
        superbasis = Superbasis.from_pymatgen_structure(struct)
        return self.from_motif_and_superbasis(motif, superbasis)