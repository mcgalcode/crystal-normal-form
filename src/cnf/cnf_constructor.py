import numpy as np
from pymatgen.core import Structure # Or other library
from .lattice import Superbasis
from .motif.atomic_motif import FractionalMotif, DiscretizedMotif
from .lattice.lnf_constructor import LatticeNormalFormConstructor, LatticeNormalFormConstructionResult, VonormSorter
from .lattice.voronoi import VonormList
from .lattice.selling import VonormListSellingReducer
from .lattice.lattice_normal_form import LatticeNormalForm
from .motif.basis_normal_form import BasisNormalForm
from .lattice.permutations import MatrixTuple
from .lattice.rounding import DiscretizedVonormComputer
from .motif.bnf_constructor import BNFConstructor, BNFConstructionResult
from .crystal_normal_form import CrystalNormalForm
from .lattice.unimodular import combine_unimodular_matrices

class CNFConstructionResult():

    def __init__(self,
                 cnf: CrystalNormalForm,
                 lnf_construction_result: LatticeNormalFormConstructionResult,
                 bnf_construction_result: BNFConstructionResult):
        self.cnf = cnf
        self.lnf_result = lnf_construction_result
        self.bnf_result = bnf_construction_result
    
    def print_details(self):
        self.lnf_result.print_details()
        print()
        self.bnf_result.print_details()

class CNFConstructor():

    def __init__(self,
                 xi: float,
                 delta: int,
                 verbose_logging: bool = False):
        self.xi = xi
        self.delta = delta
        self.verbose_logging = verbose_logging


    def from_cnf(self, cnf: CrystalNormalForm):
        assert cnf.xi == self.xi
        assert cnf.delta == self.delta
        disc_vns = cnf.lattice_normal_form.vonorms
        motif = cnf.basis_normal_form.to_discretized_motif()
        return self.from_discretized_obtuse_vonorms_and_motif(disc_vns, motif)

    def from_motif_and_superbasis(self, motif: FractionalMotif, superbasis: Superbasis):
        lnf_constructor = LatticeNormalFormConstructor(self.xi, self.verbose_logging)
        lnf_construction_result = lnf_constructor.build_lnf_from_superbasis(superbasis)
        if self.verbose_logging:
            print(f"Successfully constructed LNF! {lnf_construction_result.lnf}")
            print()
        
        res = self._from_lnf_construction_result(motif, lnf_construction_result)
        return self.from_cnf(res.cnf)
        
    def from_discretized_obtuse_vonorms_and_motif(self,
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
        if self.verbose_logging:
            print(f"Successfully constructed LNF! {lnf_construction_result.lnf}")
        return self._from_lnf_construction_result(motif, lnf_construction_result)

    def _from_lnf_construction_result(self, motif: FractionalMotif, lnf_construction_result: LatticeNormalFormConstructionResult):
        transform = lnf_construction_result.transform_mat
        motif = motif.apply_unimodular(transform)
        stabilizer_perms = lnf_construction_result.discretized_canonicalization_result.equivalent_transformations
        stabilizer_mats = [m for p in stabilizer_perms for m in p.all_matrices]
        # motif = motif.apply_unimodular(lnf_construction_result.discretized_canonicalization_result.equivalent_transformations[0].matrix)
        # stabilizer_mats = lnf_construction_result.stabilizer()

        if self.verbose_logging:
            print(f"Found {len(stabilizer_mats)} stabilizers...")

        bnf_constructor = BNFConstructor(self.delta, stabilizer_mats, self.verbose_logging)
        bnf_construction_res = bnf_constructor.build(motif)

        if self.verbose_logging:
            print(f"Found BNF! {bnf_construction_res.bnf}")
            print(f"Achieved by matrix: {bnf_construction_res.sorted_bnf_candidates[0].unimodular}")
            print(f"And shift {bnf_construction_res.sorted_bnf_candidates[0].shift}")
            print(f"Based on motif:")
            bnf_construction_res.sorted_bnf_candidates[0].motif.print_details()

        cnf = CrystalNormalForm(lnf_construction_result.lnf, bnf_construction_res.bnf)
        return CNFConstructionResult(cnf, lnf_construction_result, bnf_construction_res)
    
    def from_motif_and_basis_vecs(self, motif: FractionalMotif, basis_vecs: np.array):
        superbasis = Superbasis.from_generating_vecs(basis_vecs)
        return self.from_motif_and_superbasis(motif, superbasis)

    def from_pymatgen_structure(self, struct: Structure):
        motif = FractionalMotif.from_pymatgen_structure(struct)
        superbasis = Superbasis.from_pymatgen_structure(struct)
        return self.from_motif_and_superbasis(motif, superbasis)