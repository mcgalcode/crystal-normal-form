import numpy as np
from pymatgen.core import Structure # Or other library
from .lattice import Superbasis
from .motif.atomic_motif import FractionalMotif, DiscretizedMotif
from .lattice.lnf_constructor import LatticeNormalFormConstructor, LatticeNormalFormConstructionResult
from .lattice.voronoi import VonormList
from .lattice.permutations import MatrixTuple
from .motif.bnf_constructor import BNFConstructor, BNFConstructionResult, PreTransform
from .crystal_normal_form import CrystalNormalForm

class CNFConstructionResult():

    def __init__(self,
                 lnf_construction_result: LatticeNormalFormConstructionResult,
                 bnf_construction_result: BNFConstructionResult):
        self.lnf_result = lnf_construction_result
        self.bnf_result = bnf_construction_result
    
    @property
    def cnf(self) -> CrystalNormalForm:
        return CrystalNormalForm(self.lnf_result.lnf, self.bnf_result.bnf)
    
    def print_details(self):
        self.lnf_result.print_details()
        print()
        self.bnf_result.print_details()

class CNFConstructor():

    def __init__(self,
                 xi: float,
                 delta: int,
                 verbose_logging: bool = False,
                 extra_pretransforms: list[list[MatrixTuple]] = None):
        self.xi = xi
        self.delta = delta
        self.verbose_logging = verbose_logging
        if extra_pretransforms is None:
            extra_pretransforms = []
        else:
            extra_pretransforms = [PreTransform(mats) for mats in extra_pretransforms]
        self.extra_pretransforms = extra_pretransforms

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
        
        undisc_sort_transforms = [m.matrix for m in undisc_result.equivalent_transformations]
        disc_sort_transforms = [m.matrix for m in disc_result.equivalent_transformations]

        pre_transforms = self._get_pretransforms([
            [undisc_result.selling_transform_mat],
            undisc_sort_transforms,
            [disc_result.selling_transform_mat],
            disc_sort_transforms,
            disc_result.canonical_vonorms.stabilizer_matrices()
        ])

        # stabilizer = 

        bnf_constructor = BNFConstructor(pre_transforms)
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
        lnf_construction_result = lnf_constructor.build_lnf_from_discretized_vonorms(discretized_vonorms, skip_reduction=True)
        lnf = lnf_construction_result.lnf

        if self.verbose_logging:
            print(f"Successfully constructed LNF! {lnf}")

        pre_transforms = self._get_pretransforms([
            lnf_construction_result.discretized_canonicalization_result.equivalent_transformations[0].matrix
        ])

        stabilizer = lnf_construction_result.discretized_canonicalization_result.canonical_vonorms.stabilizer_matrices()

        bnf_constructor = BNFConstructor(pre_transforms, stabilizer)
        bnf_construction_res = bnf_constructor.build(motif)

        if self.verbose_logging:
            print(f"Found BNF! {bnf_construction_res.bnf}")
        
        return CNFConstructionResult(lnf_construction_result, bnf_construction_res)

    def _get_pretransforms(self, pretransforms):
        return [
            *self.extra_pretransforms,
            *[PreTransform(p) for p in pretransforms]
        ]

    def from_motif_and_basis_vecs(self, motif: FractionalMotif, basis_vecs: np.array):
        superbasis = Superbasis.from_generating_vecs(basis_vecs)
        return self.from_motif_and_superbasis(motif, superbasis)

    def from_pymatgen_structure(self, struct: Structure):
        motif = FractionalMotif.from_pymatgen_structure(struct)
        superbasis = Superbasis.from_pymatgen_structure(struct)
        return self.from_motif_and_superbasis(motif, superbasis)